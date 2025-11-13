#!/usr/bin/env python3
"""
YouTube Livestream Summarizer

Automatically records and summarizes YouTube livestreams using FFmpeg and Gemini API.
"""

import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path
from dotenv import load_dotenv
import schedule
import google.genai as genai
from google.genai import types

# Configuration Constants
VIDEO_DURATION_SECONDS = 120  # Duration of video clips to send to Gemini (in seconds)
SEGMENT_DURATION = 30  # Duration of each video segment (in seconds)
NUM_SEGMENTS = VIDEO_DURATION_SECONDS // SEGMENT_DURATION  # Number of segments needed
OVERLAP_SEGMENTS = 0  # Number of overlapping segments between cycles
OVERLAP_SECONDS = OVERLAP_SEGMENTS * SEGMENT_DURATION  # Duration of overlap (calculated)

# Validate segment duration to prevent Gemini rate limit issues
if SEGMENT_DURATION < 60:
    print(f"❌ Error: SEGMENT_DURATION must be at least 60 seconds to avoid Gemini API rate limits.")
    print(f"Current value: {SEGMENT_DURATION} seconds")
    print(f"Please change SEGMENT_DURATION in the script to 60 or more.")
    sys.exit(1)

# Retry Configuration
FFMPEG_MAX_RETRIES = 3  # Number of retries for FFmpeg operations (concatenation, compression)
FFMPEG_RETRY_DELAY = 120  # Seconds to wait between FFmpeg retries (2 minutes)
GEMINI_MAX_RETRIES = 3  # Number of retries for Gemini API calls
GEMINI_RETRY_DELAY = 30  # Seconds to wait between Gemini retries

# Gemini Configuration
USE_GOOGLE_SEARCH = False  # Enable/disable Google Search grounding tool in Gemini

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)

class LivestreamSummarizer:
    def __init__(self, hls_url: str, api_key: str, stream_name: str = None, custom_prompt: str = None):
        self.hls_url = hls_url
        self.api_key = api_key
        self.stream_name = stream_name or "stream"
        
        # Default prompt if none provided
        self.custom_prompt = custom_prompt or """liveposting, summary detail this stream for me in english
              1. paragraph style, no bullets style
              2. only provide liveposting nothing else, don't talk about you or something else outside the stream
              3. don't mention timestamp of the video
              4. simple english"""
        
        # Validate overlap configuration to prevent infinite loops
        global OVERLAP_SEGMENTS
        if OVERLAP_SEGMENTS >= NUM_SEGMENTS:
            logger.warning(f"OVERLAP_SEGMENTS ({OVERLAP_SEGMENTS}) >= NUM_SEGMENTS ({NUM_SEGMENTS}) would cause infinite reuse of segments!")
            logger.warning(f"Automatically adjusting OVERLAP_SEGMENTS to {NUM_SEGMENTS - 1} to prevent issues.")
            OVERLAP_SEGMENTS = NUM_SEGMENTS - 1
            # Recalculate OVERLAP_SECONDS based on new OVERLAP_SEGMENTS
            global OVERLAP_SECONDS
            OVERLAP_SECONDS = OVERLAP_SEGMENTS * SEGMENT_DURATION
        
        # Use absolute paths based on script location
        script_dir = Path(__file__).parent
        self.segments_dir = script_dir / "segments"
        self.segments_dir.mkdir(exist_ok=True)
        self.concat_file = self.segments_dir / "concat.txt"
        self.last10_file = script_dir / "last10.mp4"
        self.compressed_file = script_dir / "compressed.mp4"

        # Check if first run and clean up old files
        first_run_flag = script_dir / ".first_run"
        if not first_run_flag.exists():
            logger.info("First run detected, cleaning up old files...")
            # Clean up segments directory completely
            if self.segments_dir.exists():
                import shutil
                shutil.rmtree(self.segments_dir)
                logger.info("Removed entire segments directory")
            self.segments_dir.mkdir()
            logger.info("Recreated segments directory")
            
            # Clean up old summary txt files in main directory (both old formats)
            for txt_file in script_dir.glob("summary*.txt"):
                try:
                    txt_file.unlink()
                    logger.info(f"Deleted old summary: {txt_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {txt_file.name}: {e}")
            
            # Clean up video files
            for video_file in [self.last10_file, self.compressed_file]:
                if video_file.exists():
                    try:
                        video_file.unlink()
                        logger.info(f"Deleted old video file: {video_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {video_file.name}: {e}")
            
            # Create first run flag
            first_run_flag.touch()
            logger.info("Cleanup completed, first run flag created")

        # Configure Gemini
        self.client = genai.Client(api_key=self.api_key)

        # FFmpeg processes
        self.recording_process = None
        
        # Overlap tracking
        self.last_end_index = -1  # Index of last segment used in previous cycle
        
        # Processing flag to prevent concurrent cycles
        self.processing = False

    def start_recording(self):
        """Start FFmpeg to record segments with compression applied during recording."""
        # Clean up segments directory before starting recording
        logger.info("Cleaning segments directory before recording...")
        if self.segments_dir.exists():
            for file in self.segments_dir.glob("*"):
                try:
                    file.unlink()
                    logger.info(f"Deleted leftover file: {file.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {file.name}: {e}")
        
        # Clean up leftover video files
        for video_file in [self.last10_file, self.compressed_file]:
            if video_file.exists():
                try:
                    video_file.unlink()
                    logger.info(f"Deleted leftover video file: {video_file.name}")
                except Exception as e:
                    logger.warning(f"Could not delete {video_file.name}: {e}")
        
        logger.info("Segments directory cleaned")

        # FFmpeg command to create compressed sequential segments during recording
        # This eliminates the need for separate compression step later
        cmd = [
            'ffmpeg',
            # HLS input options for stability and reconnection
            '-reconnect', '1',              # Enable automatic reconnection
            '-reconnect_streamed', '1',     # Reconnect to streamed (HLS) input
            '-reconnect_delay_max', '10',   # Max delay between reconnects (10 seconds)
            '-timeout', '30000000',         # 30 second timeout (in microseconds)
            '-i', self.hls_url,
            '-f', 'segment',
            '-segment_time', str(SEGMENT_DURATION),
            '-segment_wrap', '0',  # Never wrap segment numbers (allows unlimited segments)
            '-reset_timestamps', '1',  # Reset timestamps for each segment
            '-c:v', 'h264_nvenc',  # NVIDIA hardware-accelerated H.264 (faster than H.265)
            '-preset', 'fast',  # Fast preset for maximum speed (ultrafast not available for NVENC)
            '-rc', 'cbr',  # Constant bitrate for consistent speed
            '-b:v', '500k',  # Lower bitrate for faster encoding (quality doesn't matter)
            '-maxrate', '500k',  # Same as target for CBR
            '-bufsize', '500k',  # Smaller buffer for speed
            '-vf', 'scale=-2:720,fps=30',  # Keep scaling for smaller files
            '-c:a', 'aac',
            '-b:a', '64k',  # Lower audio bitrate for speed
            '-movflags', '+faststart',  # Enable progressive download
            str(self.segments_dir / 'segment_%03d.mp4')
        ]
        logger.info("Starting FFmpeg recording process with compression...")
        logger.info(f"Recording compressed segments to {self.segments_dir} ({SEGMENT_DURATION}s each)")
        logger.info("Compression settings: 720p H.264 @ 500k CBR video + 64k audio (optimized for speed)")
        self.recording_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # Wait for segments to start
        logger.info("Recording started successfully")

    def validate_segment(self, segment_path, log_on_success=True, log_on_failure=True):
        """Validate that a segment file is playable using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_format',
                '-show_streams',
                str(segment_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if log_on_success:
                    logger.info(f"Segment {segment_path.name} is valid")
                return True
            else:
                if log_on_failure:
                    logger.error(f"Segment {segment_path.name} is invalid: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            if log_on_failure:
                logger.error(f"ffprobe timed out for {segment_path.name}")
            return False
        except Exception as e:
            if log_on_failure:
                logger.error(f"Error validating {segment_path.name}: {e}")
            return False

    def stop_recording(self):
        """Stop the recording process."""
        if self.recording_process:
            logger.info("Stopping FFmpeg recording process...")
            self.recording_process.terminate()
            self.recording_process.wait()
            time.sleep(2)  # Wait for files to be fully written
            logger.info("Recording stopped.")
        else:
            logger.info("No recording process to stop.")

    def wait_for_segment_completion(self, segment_path: Path, timeout=None):
        """Wait until a specific segment stops growing and passes validation."""
        if timeout is None:
            timeout = SEGMENT_DURATION * 3  # Increased to 3x segment duration (60 seconds)

        start_time = time.time()
        last_size = -1
        stable_checks = 0

        while time.time() - start_time < timeout:
            if not segment_path.exists():
                time.sleep(0.5)
                continue

            size = segment_path.stat().st_size

            if size == 0:
                stable_checks = 0
            elif size == last_size:
                stable_checks += 1
            else:
                stable_checks = 1
                last_size = size
                logger.info(f"Waiting for {segment_path.name} to finish writing ({size} bytes)")

            if size > 0 and stable_checks >= 3:
                if self.validate_segment(segment_path, log_on_success=False, log_on_failure=False):
                    logger.info(f"Segment {segment_path.name} finalized ({size} bytes)")
                    return True
                stable_checks = 0

            time.sleep(0.5)

        logger.error(f"Timed out waiting for {segment_path.name} to finish writing")
        return False

    def create_concat_file(self):
        """Create concat file with segments for current cycle, using overlap."""
        # Clean up old segments from previous cycle BEFORE processing new cycle
        # This ensures old segments are not in use when deleted
        if self.last_end_index != -1:
            self.cleanup_old_segments()
        
        segments = sorted(self.segments_dir.glob('segment_*.mp4'))
        logger.info(f"Found {len(segments)} segments in {self.segments_dir}")
        
        if not segments:
            logger.warning("No segments found")
            return False

        # Get the highest segment index available
        max_index = -1
        for seg in segments:
            try:
                idx = int(seg.stem.split('_')[1])
                max_index = max(max_index, idx)
            except (ValueError, IndexError):
                continue
        
        if max_index == -1:
            logger.warning("Could not determine segment indices")
            return False

        # Determine segment indices for this cycle
        if self.last_end_index == -1:
            # First cycle: use the latest NUM_SEGMENTS
            if max_index < NUM_SEGMENTS - 1:
                logger.warning(f"Only segments up to index {max_index} available, need up to {NUM_SEGMENTS - 1}")
                return False
            start_index = max_index - NUM_SEGMENTS + 1
            end_index = max_index
        else:
            # Subsequent cycles: start from overlap position
            start_index = max(0, self.last_end_index - OVERLAP_SEGMENTS + 1)
            end_index = start_index + NUM_SEGMENTS - 1
            
            if max_index < end_index:
                logger.warning(f"Not enough new segments for overlapping cycle. Need up to index {end_index}, have {max_index}")
                return False

        # OPTIMIZATION: Instead of waiting for last segment to complete,
        # wait for NEXT segment to start (proves all cycle segments are complete)
        next_segment_index = end_index + 1
        next_segment_path = self.segments_dir / f"segment_{next_segment_index:03d}.mp4"
        
        logger.info(f"Waiting for next segment {next_segment_path.name} to start (ensures cycle segments are complete)...")
        wait_start = time.time()
        wait_timeout = SEGMENT_DURATION * 2  # Max 40 seconds wait
        last_cycle_size = 0
        
        while time.time() - wait_start < wait_timeout:
            if next_segment_path.exists() and next_segment_path.stat().st_size > 0:
                print()  # New line after the finalizing status
                logger.info(f"Next segment {next_segment_path.name} started, cycle segments are complete")
                break
            
            # Update last segment size in real-time
            last_segment_path = self.segments_dir / f"segment_{end_index:03d}.mp4"
            if last_segment_path.exists():
                try:
                    current_size = last_segment_path.stat().st_size
                    if current_size != last_cycle_size:
                        last_cycle_size = current_size
                        size_mb = current_size / (1024 * 1024)
                        print(f"\r🔄 Finalizing: {last_segment_path.name} ({size_mb:.1f} MB)", end='', flush=True)
                except OSError:
                    pass
            
            time.sleep(0.5)
        else:
            print()  # New line after the finalizing status
            logger.warning(f"Timeout waiting for {next_segment_path.name}, proceeding with validation...")
            # Fallback: validate the last segment in cycle
            last_segment_path = self.segments_dir / f"segment_{end_index:03d}.mp4"
            if not self.wait_for_segment_completion(last_segment_path, timeout=SEGMENT_DURATION):
                logger.error(f"Failed to validate {last_segment_path.name}")
                return False

        # Extract the segments for this cycle
        cycle_segments = []
        for i in range(start_index, end_index + 1):
            seg_path = self.segments_dir / f"segment_{i:03d}.mp4"
            if seg_path.exists():
                cycle_segments.append(seg_path)
            else:
                logger.warning(f"Missing segment: {seg_path.name}")
                return False
        
        logger.info(f"Cycle segments: indices {start_index} to {end_index} ({len(cycle_segments)} segments)")

        # Validate each segment
        valid_segments = []
        for seg in cycle_segments:
            if self.validate_segment(seg):
                valid_segments.append(seg)
            else:
                logger.warning(f"Skipping invalid segment: {seg.name}")
        
        if len(valid_segments) != len(cycle_segments):
            logger.warning(f"Some segments invalid. Expected {len(cycle_segments)}, got {len(valid_segments)}")
            return False
        
        latest_segments = valid_segments
        
        # Update tracking
        self.last_end_index = end_index
        
        # Check if segments have content and log sizes
        for seg in latest_segments:
            size = seg.stat().st_size
            logger.info(f"Segment {seg.name}: {size} bytes")
        
        empty_segments = [seg for seg in latest_segments if seg.stat().st_size == 0]
        if empty_segments:
            logger.warning(f"Found {len(empty_segments)} empty segments: {[s.name for s in empty_segments]}")
        
        logger.info(f"Using segments {Path(latest_segments[0]).name} to {Path(latest_segments[-1]).name} for concatenation")
        with open(self.concat_file, 'w') as f:
            for seg in latest_segments:
                f.write(f"file '{seg}'\n")
        logger.info(f"Concat file created: {self.concat_file}")
        with open(self.concat_file, 'r') as f:
            logger.info(f"Concat file contents:\n{f.read()}")
        return True

    def concatenate_segments(self):
        """Concatenate pre-compressed segments into final video with retry logic.
        
        Segments are already compressed during recording, so we just concatenate them.
        Output goes to compressed_file directly (skipping the intermediate last10_file).
        """
        if not self.create_concat_file():
            return False

        # Optimization: Skip concatenation if only 1 segment
        if NUM_SEGMENTS == 1:
            logger.info("Single segment detected, skipping concatenation (copying directly)")
            # Read the single segment path from concat file
            with open(self.concat_file, 'r') as f:
                line = f.readline().strip()
                # Extract path from "file 'path'" format
                segment_path = Path(line.split("'")[1])
            
            try:
                import shutil
                # Copy directly to compressed_file since segments are already compressed
                shutil.copy2(segment_path, self.compressed_file)
                logger.info(f"Copied segment to {self.compressed_file} ({self.compressed_file.stat().st_size} bytes)")
                return True
            except Exception as e:
                logger.error(f"Failed to copy segment: {e}")
                return False

        max_retries = FFMPEG_MAX_RETRIES
        retry_delay = FFMPEG_RETRY_DELAY
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Starting concatenation of {NUM_SEGMENTS} pre-compressed segments into {self.compressed_file} (attempt {attempt}/{max_retries})")
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if exists
                '-f', 'concat',
                '-safe', '0',
                '-i', str(self.concat_file),
                '-c', 'copy',  # Just copy, no re-encoding needed
                str(self.compressed_file)  # Output directly to compressed_file
            ]
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            logger.info("Running FFmpeg concatenation (no re-encoding)...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"Concatenation completed successfully. Video file: {self.compressed_file} ({self.compressed_file.stat().st_size} bytes)")
                    return True
                else:
                    logger.warning(f"Concatenation attempt {attempt} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg concatenation timed out after 300 seconds (attempt {attempt})")
            except Exception as e:
                logger.warning(f"Concatenation attempt {attempt} error: {e}")
            
            # Retry logic
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Concatenation failed after {max_retries} attempts")
                return False
        
        return False

    def summarize_with_gemini(self):
        """Send video to Gemini and get summary."""
        if not self.compressed_file.exists():
            error_msg = "Compressed video file not found."
            logger.error(error_msg)
            print(f"❌ GEMINI ERROR: {error_msg}")
            return None

        try:
            logger.info(f"Uploading compressed video file to Gemini ({self.compressed_file.stat().st_size} bytes)...")
            # Upload the video file with timeout
            result = [None]
            def upload_task():
                try:
                    result[0] = self.client.files.upload(file=self.compressed_file)
                except Exception as e:
                    result[0] = e

            upload_thread = threading.Thread(target=upload_task)
            upload_thread.start()
            upload_thread.join(timeout=300)  # 5 minutes timeout

            if upload_thread.is_alive():
                error_msg = "Upload timed out after 300 seconds"
                logger.error(error_msg)
                print(f"❌ GEMINI ERROR: {error_msg}")
                return None

            if isinstance(result[0], Exception):
                raise result[0]

            video_file = result[0]
            logger.info(f"Video uploaded successfully. File URI: {video_file.uri}")

            # Poll until the file is ACTIVE
            logger.info("Waiting for file to become ACTIVE...")
            poll_count = 0
            while video_file.state.name == "PROCESSING":
                time.sleep(5)
                video_file = self.client.files.get(name=video_file.name)
                poll_count += 1
                if poll_count % 6 == 0:  # Log every 30 seconds
                    logger.info(f"File still processing... (checked {poll_count} times)")

            if video_file.state.name != "ACTIVE":
                error_msg = f"File upload failed or did not become ACTIVE: {video_file.state.name}"
                logger.error(error_msg)
                print(f"❌ GEMINI ERROR: {error_msg}")
                return None

            logger.info("File is now ACTIVE, generating summary with Gemini...")
            
            # Configure tools based on USE_GOOGLE_SEARCH setting
            if USE_GOOGLE_SEARCH:
                logger.info("Google Search grounding enabled")
                grounding_tool = types.Tool(
                    google_search=types.GoogleSearch()
                )
                config = types.GenerateContentConfig(
                    tools=[grounding_tool]
                )
            else:
                logger.info("Google Search grounding disabled")
                config = types.GenerateContentConfig()
            
            # Use custom prompt from instance variable
            logger.info(f"Using prompt: {self.custom_prompt[:50]}...")
            
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_text(text=self.custom_prompt),
                    types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
                ],
                config=config
            )
            summary = response.text
            if summary and summary.strip():
                logger.info("Summary generated successfully")
                return summary
            else:
                error_msg = "Summary response was empty or invalid"
                logger.error(error_msg)
                print(f"❌ GEMINI ERROR: {error_msg}")
                # Try to extract more details from response object
                if hasattr(response, 'prompt_feedback'):
                    logger.error(f"   Prompt feedback: {response.prompt_feedback}")
                    print(f"   Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    for idx, candidate in enumerate(response.candidates):
                        logger.error(f"   Candidate {idx}: {candidate}")
                        print(f"   Candidate {idx}: {candidate}")
                        if hasattr(candidate, 'finish_reason'):
                            logger.error(f"   Finish reason: {candidate.finish_reason}")
                            print(f"   Finish reason: {candidate.finish_reason}")
                        if hasattr(candidate, 'safety_ratings'):
                            logger.error(f"   Safety ratings: {candidate.safety_ratings}")
                            print(f"   Safety ratings: {candidate.safety_ratings}")
                return None
        except Exception as e:
            # Extract detailed error information
            error_msg = str(e)
            logger.error(f"Gemini summarization failed: {error_msg}")
            print(f"❌ GEMINI ERROR: {error_msg}")
            
            # Try to extract additional error details
            if hasattr(e, 'status_code'):
                logger.error(f"   HTTP status code: {e.status_code}")
                print(f"   HTTP status code: {e.status_code}")
            
            if hasattr(e, 'reason'):
                logger.error(f"   Reason: {e.reason}")
                print(f"   Reason: {e.reason}")
            
            if hasattr(e, 'message'):
                logger.error(f"   Message: {e.message}")
                print(f"   Message: {e.message}")
            
            if hasattr(e, 'details'):
                logger.error(f"   Details: {e.details}")
                print(f"   Details: {e.details}")
            
            # Try to get error info from response if available
            if hasattr(e, 'response'):
                try:
                    import json
                    error_info = e.response.json() if hasattr(e.response, 'json') else {}
                    if error_info:
                        logger.error(f"   Error info: {json.dumps(error_info, indent=2)}")
                        print(f"   Error info: {json.dumps(error_info, indent=2)}")
                except Exception:
                    pass
            
            return None

    def get_summary_number(self):
        """Get the next sequential summary number by counting existing entries."""
        summary_file = Path(f"summary-{self.stream_name}.txt")
        
        if not summary_file.exists():
            return 1
        
        # Count how many summary entries exist by counting '#' markers
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count lines that start with '#' followed by a digit
                import re
                matches = re.findall(r'^#(\d+)', content, re.MULTILINE)
                if matches:
                    return max(int(m) for m in matches) + 1
                return 1
        except Exception:
            return 1
    
    def append_summary(self, summary_text, timestamp):
        """Append a new summary to the single summary file with stream name."""
        summary_file = Path(f"summary-{self.stream_name}.txt")
        summary_num = self.get_summary_number()
        
        # Prepare the entry
        entry = f"#{summary_num} - {timestamp}\n{summary_text}\n\n"
        
        # Append to file (create if doesn't exist)
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(entry)
        
        logger.info(f"Summary #{summary_num} appended to {summary_file.name}")
        return summary_num

    def cleanup_old_segments(self):
        """Clean up old segments to prevent unlimited disk usage.
        
        Keeps only the most recent overlapping segment from previous cycles.
        Example: After cycle using segments 0,1,2 → keeps only segment 2
                 After cycle using segments 2,3,4 → deletes 0,1, keeps 4
        """
        if self.last_end_index == -1:
            # First cycle, nothing to clean yet
            return

        try:
            # Give FFmpeg a moment to fully release file handles
            time.sleep(2)
            
            segments = sorted(self.segments_dir.glob('segment_*.mp4'))
            if not segments:
                return

            # Keep only the most recent overlapping segment
            # This is the segment that will be used in the next cycle's overlap
            keep_from_index = self.last_end_index - OVERLAP_SEGMENTS + 1
            
            segments_to_delete = []
            for segment in segments:
                # Extract segment number from filename (segment_XXX.mp4)
                try:
                    segment_num = int(segment.stem.split('_')[1])
                    if segment_num < keep_from_index:
                        segments_to_delete.append(segment)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse segment number from {segment.name}")
                    continue

            # Delete old segments
            deleted_count = 0
            for segment in segments_to_delete:
                try:
                    segment.unlink()
                    logger.info(f"Cleaned up old segment: {segment.name}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {segment.name}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old segments, keeping from index {keep_from_index} onwards")

        except Exception as e:
            logger.warning(f"Error during segment cleanup: {e}")

    def process_and_summarize(self):
        """Main processing function: concat, summarize."""
        # Check if already processing
        if self.processing:
            return
            
        logger.info("=== Starting summarization cycle ===")
        self.processing = True
        
        # Keep recording running - process summarization in background
        if not self.concatenate_segments():
            logger.info("Summarization cycle skipped due to concatenation failure")
            self.processing = False
            return

        # Start summarization in background thread to keep recording continuous
        summarization_thread = threading.Thread(target=self._background_summarization)
        summarization_thread.daemon = True
        summarization_thread.start()
        
        logger.info("=== Summarization cycle started (running in background) ===")

    def _background_summarization(self):
        """Background thread for video processing and Gemini summarization.
        
        Note: Segments are already compressed during recording, so no separate
        compression step is needed. Concatenation outputs directly to compressed_file.
        """
        try:
            logger.info("Sending pre-compressed video to Gemini for summarization...")
            
            # Retry logic for Gemini summarization
            max_retries = GEMINI_MAX_RETRIES
            retry_delay = GEMINI_RETRY_DELAY
            summary = None
            
            for attempt in range(1, max_retries + 1):
                if attempt > 1:
                    logger.info(f"Retrying summarization (attempt {attempt}/{max_retries})...")
                
                summary = self.summarize_with_gemini()
                
                if summary:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Summary received at {timestamp}")
                    print(f"\n--- Summary at {timestamp} ---\n{summary}\n")
                    # Append to single summary file with stream name
                    summary_num = self.append_summary(summary, timestamp)
                    print(f"✅ Summary #{summary_num} saved to summary-{self.stream_name}.txt")
                    break
                else:
                    # Error details already logged in summarize_with_gemini()
                    if attempt < max_retries:
                        logger.warning(f"Summarization attempt {attempt} failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Summary generation failed after {max_retries} attempts")

            # Clean up video file (compressed_file only, no last10_file anymore)
            try:
                if self.compressed_file.exists():
                    self.compressed_file.unlink()
                    logger.info(f"Deleted {self.compressed_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up video file: {e}")

            logger.info("=== Summarization cycle completed ===")
            
        except Exception as e:
            logger.error(f"Error in background summarization: {e}")
        finally:
            # Release processing flag
            self.processing = False

    def run(self):
        """Main loop."""
        self.start_recording()

        cycle_count = 0
        last_segment_count = 0
        last_size_update = 0
        current_recording_size = 0
        
        try:
            while True:
                schedule.run_pending()  # Keep for any other scheduled tasks
                
                # Check and log segment accumulation progress
                segments = list(self.segments_dir.glob('segment_*.mp4'))
                current_count = len(segments)
                
                # Calculate required max index for next cycle
                if segments:
                    try:
                        max_index = max(int(seg.stem.split('_')[1]) for seg in segments if seg.stem.split('_')[1].isdigit())
                        if self.last_end_index == -1:
                            required_max_index = NUM_SEGMENTS - 1
                        else:
                            required_max_index = self.last_end_index + NUM_SEGMENTS - OVERLAP_SEGMENTS
                        
                        if max_index >= required_max_index and not self.processing:
                            cycle_count += 1
                            logger.info(f"Starting summarization cycle #{cycle_count}")
                            self.process_and_summarize()
                    except ValueError:
                        logger.warning("Could not parse segment indices")
                
                # Log segment progress only when count changes
                if current_count != last_segment_count and current_count < NUM_SEGMENTS:
                    logger.info(f"Segments accumulated: {current_count}/{NUM_SEGMENTS} ({NUM_SEGMENTS - current_count} remaining)")
                    last_segment_count = current_count
                
                # Update current recording file size every 5 seconds
                current_time = time.time()
                if current_time - last_size_update >= 5:
                    if segments:
                        # Get the current segment being recorded (highest index)
                        try:
                            current_segment = max(segments, key=lambda s: int(s.stem.split('_')[1]))
                            new_size = current_segment.stat().st_size
                            if new_size != current_recording_size:
                                current_recording_size = new_size
                                # Update the same line with current file size
                                size_mb = current_recording_size / (1024 * 1024)
                                print(f"\r📹 Recording: {current_segment.name} ({size_mb:.1f} MB)", end='', flush=True)
                        except (ValueError, OSError):
                            pass
                    last_size_update = current_time
                
                time.sleep(1)
        except KeyboardInterrupt:
            print()  # New line after the recording status
            logger.info("Stopping...")
            self.stop_recording()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python main.py <youtube_url_or_hls_url> [prompt]")
        print("  - If prompt is empty, uses default prompt")
        print("  - If prompt is provided, uses it for Gemini summarization")
        sys.exit(1)

    url = sys.argv[1]
    custom_prompt = sys.argv[2] if len(sys.argv) == 3 else None
    stream_name = None  # Will be auto-extracted from YouTube title
    
    # Check if it's a YouTube URL and extract HLS URL using yt-dlp
    if 'youtube.com' in url or 'youtu.be' in url:
        print("Detected YouTube URL, extracting HLS stream URL...")
        
        # Extract stream name from YouTube URL if not provided
        if not stream_name:
            try:
                # Get video title to use as stream name
                title_result = subprocess.run(
                    ['yt-dlp', '--get-title', url],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if title_result.returncode == 0 and title_result.stdout.strip():
                    # Sanitize title for filename (allow Unicode characters)
                    import re
                    # Remove only characters that are problematic for filenames
                    # Keep Unicode letters, numbers, spaces, hyphens, underscores
                    stream_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title_result.stdout.strip())
                    stream_name = re.sub(r'[-\s]+', '-', stream_name)[:50]  # Limit length
                    print(f"Stream name: {stream_name}")
            except Exception as e:
                print(f"Could not get stream name: {e}")
                stream_name = "stream"
        
        try:
            # Try without format selector first
            result = subprocess.run(
                ['yt-dlp', '-g', url],
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            # If failed, try with explicit format
            if result.returncode != 0:
                print("⚠️ Retrying with explicit format...")
                result = subprocess.run(
                    ['yt-dlp', '-f', 'b', '-g', url],
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
            
            if result.returncode == 0 and result.stdout.strip():
                hls_url = result.stdout.strip()
                print(f"Got HLS URL: {hls_url}")
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"❌ Failed to extract HLS URL from YouTube")
                if "Sign in to confirm" in error_msg or "not a bot" in error_msg:
                    print("⚠️ YouTube detected bot - requires authentication")
                    print("💡 Try: pip install -U yt-dlp (update to latest)")
                else:
                    print(f"Error: {error_msg}")
                    print("Make sure yt-dlp is installed and the YouTube URL is valid.")
                sys.exit(1)
        except FileNotFoundError:
            print("❌ yt-dlp not found. Install yt-dlp with: pip install yt-dlp")
            print("Or provide the HLS URL directly.")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("❌ yt-dlp timed out. Check your internet connection.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running yt-dlp: {e}")
            sys.exit(1)
    else:
        # Assume it's already an HLS URL
        hls_url = url
        print(f"Using provided HLS URL: {hls_url}")
        if not stream_name:
            stream_name = "stream"

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY in .env file")
        sys.exit(1)

    # Show prompt being used
    if custom_prompt:
        print(f"Using custom prompt: {custom_prompt[:50]}...")
    else:
        print("Using default prompt")
    
    print(f"Summary file: summary-{stream_name}.txt")
    summarizer = LivestreamSummarizer(hls_url, api_key, stream_name, custom_prompt)
    summarizer.run()

if __name__ == "__main__":
    main()