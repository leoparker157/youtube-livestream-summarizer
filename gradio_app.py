#!/usr/bin/env python3
"""
Gradio Interface for YouTube Livestream Summarizer
Compatible with Google Colab
"""

import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)

class LivestreamSummarizerGradio:
    def __init__(self):
        self.recording_process = None
        self.ffmpeg_stderr = None  # Store FFmpeg stderr for debugging
        self.segment_stall_check_time = None  # Track when we first detected potential stall
        self.last_segment_size = 0  # Track segment size changes
        self.ffmpeg_restart_count = 0  # Track how many times we've restarted FFmpeg
        self.last_restart_time = 0  # Track when we last restarted
        self.processing = False
        self.should_stop = False
        self.last_end_index = -1
        self.progress_log = []
        self.summaries = []
        self.new_summary_available = False  # Flag for background thread updates
        
    def log_progress(self, message):
        """Add message to progress log"""
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.progress_log.append(log_entry)
        logger.info(message)
        return "\n".join(self.progress_log)
    
    def add_summary(self, summary_text):
        """Add summary to summaries list"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        summary_entry = f"📝 Summary at {timestamp}\n{'='*60}\n{summary_text}\n\n"
        self.summaries.append(summary_entry)
        return "\n".join(self.summaries)
    

    
    def check_hls_stream_health(self, hls_url):
        """Check if HLS stream is still accessible"""
        try:
            # Quick probe of the HLS stream
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams', hls_url],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def check_nvenc_available(self):
        """Check if NVIDIA NVENC encoder is available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return 'h264_nvenc' in result.stdout
        except:
            return False
    
    def start_recording(self, hls_url, segment_duration, segments_dir):
        """Start FFmpeg recording - EXACTLY matches main.py"""
        
        # Check for NVENC availability
        has_nvenc = self.check_nvenc_available()
        
        if has_nvenc:
            self.log_progress("🎮 GPU detected - using h264_nvenc hardware encoding")
            video_codec = 'h264_nvenc'
            codec_preset = 'fast'
            codec_options = [
                '-rc', 'cbr',
                '-b:v', '500k',
                '-maxrate', '500k',
                '-bufsize', '500k'
            ]
        else:
            self.log_progress("❌ ERROR: No GPU detected!")
            self.log_progress("⚠️ GPU (h264_nvenc) is REQUIRED for livestream recording")
            self.log_progress("⚠️ CPU encoding is too slow for real-time livestreams")
            self.log_progress("📝 In Google Colab: Runtime → Change runtime type → T4 GPU")
            raise RuntimeError(
                "GPU with h264_nvenc encoder is required. "
                "Please enable GPU in Colab (Runtime → Change runtime type → T4 GPU) "
                "and restart the runtime."
            )
        
        # FFmpeg command with HLS timeout and reconnect options
        cmd = [
            'ffmpeg',
            # HLS input options for stability
            '-reconnect', '1',              # Enable automatic reconnection
            '-reconnect_streamed', '1',     # Reconnect to streamed (HLS) input
            '-reconnect_delay_max', '10',   # Max delay between reconnects (10 seconds)
            '-timeout', '30000000',         # 30 second timeout (in microseconds)
            '-i', hls_url,
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-segment_wrap', '0',
            '-reset_timestamps', '1',
            '-c:v', video_codec,
            '-preset', codec_preset,
        ] + codec_options + [
            '-vf', 'scale=-2:720,fps=30',
            '-c:a', 'aac',
            '-b:a', '64k',
            '-movflags', '+faststart',
            str(segments_dir / 'segment_%03d.mp4')
        ]
        
        self.log_progress(f"⚙️ FFmpeg: 720p H.264 @ 500k CBR + 64k audio")
        # Capture stderr for debugging (but don't print to console)
        self.recording_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        time.sleep(5)
    
    def restart_ffmpeg(self, segment_duration, segments_dir):
        """Restart FFmpeg recording after detecting stall"""
        try:
            # Kill old process
            if self.recording_process and self.recording_process.poll() is None:
                self.log_progress("⏹️ Terminating stalled FFmpeg process...")
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=5)
                except:
                    self.recording_process.kill()
                    self.recording_process.wait()
            
            # Wait a moment for file handles to release
            time.sleep(2)
            
            # Delete stalled/incomplete segments (so FFmpeg can create fresh ones)
            self.log_progress("🗑️ Cleaning up stalled segments...")
            segments = list(segments_dir.glob('segment_*.mp4'))
            if segments:
                # Get the current/latest segment (the stalled one)
                try:
                    max_index = max(int(seg.stem.split('_')[1]) for seg in segments if seg.stem.split('_')[1].isdigit())
                    stalled_segment = segments_dir / f"segment_{max_index:03d}.mp4"
                    
                    if stalled_segment.exists():
                        size_mb = stalled_segment.stat().st_size / (1024 * 1024)
                        stalled_segment.unlink()
                        self.log_progress(f"   Deleted stalled {stalled_segment.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    self.log_progress(f"   ⚠️ Could not delete stalled segment: {e}")
            
            # Re-extract HLS URL (might have expired)
            self.log_progress("🔄 Re-extracting fresh HLS URL from YouTube...")
            result = subprocess.run(
                ['yt-dlp', '-g', self.youtube_url],
                capture_output=True, text=True, timeout=30
            )
            
            # Try with explicit format if failed
            if result.returncode != 0:
                result = subprocess.run(
                    ['yt-dlp', '-f', 'b', '-g', self.youtube_url],
                    capture_output=True, text=True, timeout=30
                )
            
            if result.returncode == 0 and result.stdout.strip():
                new_hls_url = result.stdout.strip()
                self.log_progress("✅ Got fresh HLS URL")
            else:
                self.log_progress("❌ Could not extract fresh HLS URL")
                return False
            
            # Restart recording
            self.log_progress("🎬 Restarting FFmpeg recording...")
            self.start_recording(new_hls_url, segment_duration, segments_dir)
            
            self.ffmpeg_restart_count += 1
            self.last_restart_time = time.time()
            self.segment_stall_check_time = None  # Reset stall detection
            self.last_segment_size = 0
            
            self.log_progress(f"✅ FFmpeg restarted (restart #{self.ffmpeg_restart_count})")
            return True
            
        except Exception as e:
            self.log_progress(f"❌ FFmpeg restart failed: {e}")
            logger.error(f"Restart error: {e}")
            return False
    
    def validate_segment(self, segment_path):
        """Validate segment file"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', str(segment_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def concatenate_segments(self, segments_dir, concat_file, compressed_file, segment_files):
        """Concatenate segments with validation and retry logic (like main.py)"""
        
        # Verify all segments exist
        missing_segments = [seg for seg in segment_files if not seg.exists()]
        if missing_segments:
            logger.error(f"Missing segments: {[s.name for s in missing_segments]}")
            return False
        
        # Validate each segment
        valid_segments = []
        for seg in segment_files:
            if self.validate_segment(seg):
                valid_segments.append(seg)
            else:
                logger.warning(f"Invalid segment: {seg.name}")
        
        if len(valid_segments) != len(segment_files):
            logger.warning(f"Some segments invalid. Expected {len(segment_files)}, got {len(valid_segments)}")
            return False
        
        # Check for empty segments
        empty_segments = [seg for seg in segment_files if seg.stat().st_size == 0]
        if empty_segments:
            logger.warning(f"Found {len(empty_segments)} empty segments: {[s.name for s in empty_segments]}")
            return False
        
        # Log segment sizes
        for seg in segment_files:
            size = seg.stat().st_size
            logger.info(f"Segment {seg.name}: {size} bytes")
        
        # Create concat file with specific segments
        with open(concat_file, 'w') as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")
        
        logger.info(f"Concat file created: {concat_file}")
        with open(concat_file, 'r') as f:
            logger.info(f"Concat file contents:\n{f.read()}")
        
        # Concatenate with retry logic (like main.py)
        max_retries = 3
        retry_delay = 120
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Starting concatenation (attempt {attempt}/{max_retries})")
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file), '-c', 'copy', str(compressed_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"Concatenation completed successfully")
                    return True
                else:
                    logger.warning(f"Concatenation attempt {attempt} failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg concatenation timed out (attempt {attempt})")
            except Exception as e:
                logger.warning(f"Concatenation attempt {attempt} error: {e}")
            
            # Retry logic
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error(f"Concatenation failed after {max_retries} attempts")
        return False
    
    def summarize_with_gemini(self, compressed_file, api_key, prompt_text, model_name, use_google_search=False):
        """Send video to Gemini and get summary"""
        try:
            client = genai.Client(api_key=api_key)
            
            # Upload video
            self.log_progress("📤 Uploading video to Gemini...")
            video_file = client.files.upload(file=compressed_file)
            
            # Wait for processing
            self.log_progress("⏳ Waiting for Gemini to process video...")
            poll_count = 0
            while video_file.state.name == "PROCESSING":
                time.sleep(5)
                video_file = client.files.get(name=video_file.name)
                poll_count += 1
                if poll_count % 6 == 0:  # Log every 30 seconds
                    self.log_progress(f"⏳ Still processing... ({poll_count * 5}s elapsed)")
            
            if video_file.state.name != "ACTIVE":
                self.log_progress(f"❌ Video upload failed: {video_file.state.name}")
                return None
            
            self.log_progress("✅ Video uploaded and active")
            
            # Generate summary with optional Google Search
            if use_google_search:
                self.log_progress(f"🔍 Generating summary with {model_name} (Google Search enabled)...")
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(tools=[grounding_tool])
            else:
                self.log_progress(f"🤖 Generating summary with {model_name}...")
                config = types.GenerateContentConfig()
            
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_text(text=prompt_text),
                    types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
                ],
                config=config
            )
            
            if response.text:
                self.log_progress("✅ Summary generated successfully")
                return response.text
            else:
                self.log_progress("❌ Summary was empty")
                return None
        except Exception as e:
            error_msg = str(e)
            self.log_progress(f"❌ Gemini error: {error_msg}")
            logger.error(f"Gemini error details: {e}")
            return None
    
    def _background_summarization(self, params):
        """Background thread for Gemini summarization (keeps recording running)"""
        try:
            compressed_file = params['compressed_file']
            api_key = params['api_key']
            prompt_text = params['prompt_text']
            model_name = params['model_name']
            use_google_search = params['use_google_search']
            cycle_count = params['cycle_count']
            
            self.log_progress("🤖 Sending to Gemini AI (background)...")
            summary = self.summarize_with_gemini(compressed_file, api_key, prompt_text, model_name, use_google_search)
            
            if summary:
                self.log_progress(f"✅ Summary #{cycle_count} received")
                self.add_summary(summary)
                self.new_summary_available = True  # Signal new summary to main loop
            else:
                self.log_progress("❌ Summary generation failed")
            
            # Cleanup compressed file only (segments are cleaned before next cycle)
            self.log_progress("🧹 Cleaning up compressed video file...")
            try:
                if compressed_file.exists():
                    compressed_file.unlink()
                    self.log_progress("✅ Compressed file deleted")
            except Exception as e:
                logger.warning(f"Failed to delete {compressed_file}: {e}")
                self.log_progress(f"⚠️ Could not delete compressed file: {e}")
            
        except Exception as e:
            self.log_progress(f"❌ Background summarization error: {e}")
            logger.error(f"Background error: {e}")
    
    def cleanup_old_segments(self, segments_dir, overlap_segments):
        """Clean up old segments to prevent unlimited disk usage.
        
        Keeps only the most recent overlapping segments from previous cycles.
        """
        if self.last_end_index == -1:
            # First cycle, nothing to clean yet
            return

        try:
            # Give FFmpeg a moment to fully release file handles
            time.sleep(2)
            
            segments = sorted(segments_dir.glob('segment_*.mp4'))
            if not segments:
                return

            # Keep only segments from the overlap point onwards
            keep_from_index = self.last_end_index - overlap_segments + 1
            
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
                self.log_progress(f"🧹 Deleted {deleted_count} old segments (keeping from index {keep_from_index})")

        except Exception as e:
            logger.warning(f"Error during segment cleanup: {e}")
    
    def run_summarizer(self, youtube_url, api_key, video_duration, segment_duration, 
                       overlap_segments, model_name, prompt_text, use_google_search, progress=gr.Progress()):
        """Main summarization loop"""
        self.should_stop = False
        self.progress_log = []
        self.summaries = []
        self.last_end_index = -1  # Reset for new run
        self.processing = False  # Reset processing flag
        self.new_summary_available = False  # Reset summary flag
        self.segment_stall_check_time = None  # Reset stall detection
        self.last_segment_size = 0  # Reset segment size tracking
        self.ffmpeg_restart_count = 0  # Reset restart counter
        self.last_restart_time = 0  # Reset restart time
        self.youtube_url = youtube_url  # Store for FFmpeg restart
        
        # Setup directories
        script_dir = Path.cwd()
        segments_dir = script_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        concat_file = segments_dir / "concat.txt"
        compressed_file = script_dir / "compressed.mp4"
        
        # Clean up old files before starting
        yield self.log_progress("🧹 Cleaning up old files..."), ""
        for file in segments_dir.glob("*"):
            try:
                file.unlink()
                self.log_progress(f"  Deleted: {file.name}")
            except Exception as e:
                logger.warning(f"Could not delete {file.name}: {e}")
        
        if compressed_file.exists():
            try:
                compressed_file.unlink()
                self.log_progress(f"  Deleted: {compressed_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete {compressed_file.name}: {e}")
        
        yield self.log_progress("✅ Cleanup complete"), ""
        
        num_segments = video_duration // segment_duration
        
        yield self.log_progress(f"⚙️ Configuration: {video_duration}s clips, {segment_duration}s segments, {num_segments} segments per cycle"), ""
        yield self.log_progress(f"🤖 Model: {model_name} | Google Search: {'Enabled' if use_google_search else 'Disabled'}"), ""
        
        # Extract HLS URL
        yield self.log_progress("🔍 Extracting HLS URL from YouTube..."), ""
        
        try:
            # Try without format selector first (yt-dlp will pick best)
            result = subprocess.run(
                ['yt-dlp', '-g', youtube_url],
                capture_output=True, text=True, timeout=30
            )
            
            # If that fails, try with explicit format
            if result.returncode != 0:
                yield self.log_progress("⚠️ Retrying with explicit format..."), ""
                result = subprocess.run(
                    ['yt-dlp', '-f', 'b', '-g', youtube_url],
                    capture_output=True, text=True, timeout=30
                )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                yield self.log_progress(f"❌ Failed to extract HLS URL"), ""
                
                # Check for bot detection
                if "Sign in to confirm" in error_msg or "not a bot" in error_msg:
                    yield self.log_progress(f"   ⚠️ YouTube detected bot - requires authentication"), ""
                    yield self.log_progress(f"   💡 Try: pip install -U yt-dlp (update to latest)"), ""
                else:
                    yield self.log_progress(f"   Error: {error_msg}"), ""
                    yield self.log_progress(f"   Make sure yt-dlp is installed and YouTube URL is valid"), ""
                return
            hls_url = result.stdout.strip()
            if not hls_url:
                yield self.log_progress("❌ Empty HLS URL returned"), ""
                yield self.log_progress("   The stream might not be live or URL is invalid"), ""
                return
            yield self.log_progress(f"✅ HLS URL extracted"), ""
            logger.info(f"HLS URL: {hls_url}")
        except FileNotFoundError:
            yield self.log_progress(f"❌ yt-dlp not found!"), ""
            yield self.log_progress(f"   Install with: pip install yt-dlp"), ""
            return
        except subprocess.TimeoutExpired:
            yield self.log_progress(f"❌ yt-dlp timed out after 30 seconds"), ""
            yield self.log_progress(f"   Check your internet connection"), ""
            return
        except Exception as e:
            yield self.log_progress(f"❌ Error extracting HLS URL: {e}"), ""
            logger.error(f"HLS extraction error: {e}")
            return
        
        # Start recording
        yield self.log_progress("🎬 Starting FFmpeg recording..."), ""
        try:
            self.start_recording(hls_url, segment_duration, segments_dir)
            yield self.log_progress(f"✅ Recording started (segments: {segment_duration}s each)"), ""
        except Exception as e:
            yield self.log_progress(f"❌ Recording error: {e}"), ""
            return
        
        cycle_count = 0
        
        try:
            while not self.should_stop:
                # Check if FFmpeg recording process is still running
                if self.recording_process and self.recording_process.poll() is not None:
                    # FFmpeg died - capture stderr for diagnostics
                    exit_code = self.recording_process.returncode
                    error_msg = f"❌ FFmpeg process died! Exit code: {exit_code}"
                    yield self.log_progress(""), "\n".join(self.summaries)
                    yield self.log_progress("="*60), "\n".join(self.summaries)
                    yield self.log_progress(error_msg), "\n".join(self.summaries)
                    yield self.log_progress(f"🔍 Cycle: {cycle_count}"), "\n".join(self.summaries)
                    yield self.log_progress(f"🔍 Last segment index: {self.last_end_index}"), "\n".join(self.summaries)
                    
                    # Check what segments exist
                    segments = list(segments_dir.glob('segment_*.mp4'))
                    if segments:
                        segment_list = sorted([seg.name for seg in segments])
                        segment_sizes = [(seg.name, seg.stat().st_size / (1024*1024)) for seg in sorted(segments)]
                        yield self.log_progress(f"🔍 Segments on disk: {len(segments)}"), "\n".join(self.summaries)
                        for name, size in segment_sizes:
                            yield self.log_progress(f"     {name}: {size:.2f} MB"), "\n".join(self.summaries)
                    else:
                        yield self.log_progress(f"� No segments found on disk"), "\n".join(self.summaries)
                    
                    # Try to get stderr output (read whatever's available without blocking)
                    yield self.log_progress(""), "\n".join(self.summaries)
                    try:
                        if self.recording_process.stderr:
                            # Read all available stderr (non-blocking)
                            stderr_output = self.recording_process.stderr.read()
                            if stderr_output and stderr_output.strip():
                                # Get last 30 lines of error output for better context
                                error_lines = stderr_output.strip().split('\n')[-30:]
                                yield self.log_progress("📋 FFmpeg output (last 30 lines):"), "\n".join(self.summaries)
                                for line in error_lines:
                                    if line.strip():  # Skip empty lines
                                        yield self.log_progress(f"  {line}"), "\n".join(self.summaries)
                            else:
                                yield self.log_progress("📋 (FFmpeg stderr was empty)"), "\n".join(self.summaries)
                        else:
                            yield self.log_progress("📋 (FFmpeg stderr not available)"), "\n".join(self.summaries)
                    except Exception as e:
                        yield self.log_progress(f"📋 Error reading FFmpeg output: {e}"), "\n".join(self.summaries)
                    
                    # Common issue hints
                    yield self.log_progress(""), "\n".join(self.summaries)
                    yield self.log_progress("💡 Diagnosis:"), "\n".join(self.summaries)
                    yield self.log_progress("  Exit code 0 = FFmpeg thinks it finished successfully"), "\n".join(self.summaries)
                    yield self.log_progress("  This usually means the HLS stream URL EXPIRED"), "\n".join(self.summaries)
                    yield self.log_progress(""), "\n".join(self.summaries)
                    yield self.log_progress("💡 Why this happens:"), "\n".join(self.summaries)
                    yield self.log_progress("  YouTube HLS URLs expire after a few minutes"), "\n".join(self.summaries)
                    yield self.log_progress("  FFmpeg reaches 'end of stream' when URL expires"), "\n".join(self.summaries)
                    yield self.log_progress("  We need to re-extract fresh URLs periodically"), "\n".join(self.summaries)
                    yield self.log_progress(""), "\n".join(self.summaries)
                    yield self.log_progress("💡 Current workarounds:"), "\n".join(self.summaries)
                    yield self.log_progress("  1. Use shorter video durations (60-90s instead of 120s)"), "\n".join(self.summaries)
                    yield self.log_progress("  2. Restart the summarizer every ~2 minutes"), "\n".join(self.summaries)
                    yield self.log_progress("  3. Wait for automatic URL refresh feature (coming soon)"), "\n".join(self.summaries)
                    break
                
                # Check segments
                segments = list(segments_dir.glob('segment_*.mp4'))
                
                # Calculate required max index for next cycle
                can_process = False
                max_index = -1  # Initialize
                current_count = 0
                
                if segments:
                    try:
                        max_index = max(int(seg.stem.split('_')[1]) for seg in segments if seg.stem.split('_')[1].isdigit())
                        current_count = max_index + 1
                        
                        if self.last_end_index == -1:
                            # First cycle: need at least num_segments
                            required_max_index = num_segments - 1
                        else:
                            # Subsequent cycles: need num_segments beyond last overlap point
                            required_max_index = self.last_end_index + num_segments - overlap_segments
                        
                        can_process = max_index >= required_max_index and not self.processing
                    except (ValueError, StopIteration):
                        # If parsing fails, we still have segments, just can't get max index
                        current_count = len(segments)
                        max_index = -1
                
                if can_process:
                    cycle_count += 1
                    self.processing = True
                    
                    yield self.log_progress(f"📊 Cycle #{cycle_count}: Starting..."), "\n".join(self.summaries)
                    
                    # Clean up old segments from previous cycle BEFORE processing new cycle
                    if self.last_end_index != -1:
                        yield self.log_progress("🧹 Cleaning up old segments from previous cycle..."), "\n".join(self.summaries)
                        self.cleanup_old_segments(segments_dir, overlap_segments)
                    
                    # Determine segment range for this cycle (EXACTLY like main.py)
                    if self.last_end_index == -1:
                        # First cycle: use the latest num_segments
                        start_index = max_index - num_segments + 1
                        end_index = max_index
                    else:
                        # Subsequent cycles: start from overlap position
                        start_index = max(0, self.last_end_index - overlap_segments + 1)
                        end_index = start_index + num_segments - 1
                    
                    yield self.log_progress(f"📊 Cycle segments: indices {start_index} to {end_index} ({num_segments} segments)"), "\n".join(self.summaries)
                    
                    # Wait for NEXT segment to start (proves all cycle segments are complete)
                    next_segment_index = end_index + 1
                    next_segment_path = segments_dir / f"segment_{next_segment_index:03d}.mp4"
                    yield self.log_progress(f"⏳ Waiting for next segment {next_segment_path.name} to start..."), "\n".join(self.summaries)
                    
                    wait_start = time.time()
                    wait_timeout = segment_duration * 2  # 2x segment duration
                    last_cycle_size = 0
                    finalizing_stall_start = None  # Track stall during finalization
                    
                    while time.time() - wait_start < wait_timeout:
                        if next_segment_path.exists() and next_segment_path.stat().st_size > 0:
                            yield self.log_progress(f"✅ Next segment {next_segment_path.name} started, cycle segments complete"), "\n".join(self.summaries)
                            break
                        
                        # Show last segment finalizing status
                        last_segment_path = segments_dir / f"segment_{end_index:03d}.mp4"
                        if last_segment_path.exists():
                            try:
                                current_size = last_segment_path.stat().st_size
                                if current_size != last_cycle_size:
                                    # Size changed - reset stall detection
                                    last_cycle_size = current_size
                                    finalizing_stall_start = None
                                    size_mb = current_size / (1024 * 1024)
                                    yield self.log_progress(f"🔄 Finalizing: {last_segment_path.name} ({size_mb:.1f} MB)"), "\n".join(self.summaries)
                                else:
                                    # Size hasn't changed - check for stall
                                    if finalizing_stall_start is None:
                                        finalizing_stall_start = time.time()
                                    elif time.time() - finalizing_stall_start >= 3:
                                        # Stalled for 3+ seconds during finalization
                                        stall_duration = int(time.time() - finalizing_stall_start)
                                        size_mb = current_size / (1024 * 1024)
                                        yield self.log_progress(f"🚨 STALL DETECTED: {last_segment_path.name} not growing for {stall_duration}s at {size_mb:.1f} MB"), "\n".join(self.summaries)
                                        
                                        # Check restart cooldown
                                        if time.time() - self.last_restart_time < 30:
                                            yield self.log_progress(f"⚠️ Skipping restart (last restart was {int(time.time() - self.last_restart_time)}s ago)"), "\n".join(self.summaries)
                                        else:
                                            yield self.log_progress("🔄 Attempting to restart FFmpeg..."), "\n".join(self.summaries)
                                            
                                            if self.restart_ffmpeg(segment_duration, segments_dir):
                                                yield self.log_progress("✅ FFmpeg restarted successfully"), "\n".join(self.summaries)
                                                # Break out and wait for new segments
                                                break
                                            else:
                                                yield self.log_progress("❌ FFmpeg restart failed - stopping"), "\n".join(self.summaries)
                                                self.should_stop = True
                                                break
                            except OSError:
                                pass
                        
                        time.sleep(0.5)
                    else:
                        # Timeout - but continue anyway (segments should be ready)
                        yield self.log_progress(f"⏳ {next_segment_path.name} not started yet, proceeding with current segments..."), "\n".join(self.summaries)
                    
                    # Get segment files for concatenation
                    segment_files = [segments_dir / f"segment_{i:03d}.mp4" for i in range(start_index, end_index + 1)]
                    total_size = sum(seg.stat().st_size for seg in segment_files if seg.exists()) / (1024 * 1024)
                    yield self.log_progress(f"📁 Using segments: {segment_files[0].name} to {segment_files[-1].name} ({total_size:.1f} MB)"), "\n".join(self.summaries)
                    
                    # Update last_end_index for next cycle
                    self.last_end_index = end_index
                    
                    # Concatenate and start background processing (like main.py)
                    yield self.log_progress("🔗 Concatenating segments..."), "\n".join(self.summaries)
                    if self.concatenate_segments(segments_dir, concat_file, compressed_file, segment_files):
                        final_size = compressed_file.stat().st_size / (1024 * 1024)
                        yield self.log_progress(f"✅ Concatenation complete ({final_size:.1f} MB)"), "\n".join(self.summaries)
                        
                        # Start background summarization thread (keeps recording running)
                        yield self.log_progress("🚀 Starting background summarization (recording continues)..."), "\n".join(self.summaries)
                        
                        # Store parameters for background thread
                        bg_params = {
                            'compressed_file': compressed_file,
                            'api_key': api_key,
                            'prompt_text': prompt_text,
                            'model_name': model_name,
                            'use_google_search': use_google_search,
                            'cycle_count': cycle_count
                        }
                        
                        # Start background thread
                        bg_thread = threading.Thread(target=self._background_summarization, args=(bg_params,))
                        bg_thread.daemon = True
                        bg_thread.start()
                        
                        yield self.log_progress("✅ Summarization running in background"), "\n".join(self.summaries)
                    else:
                        yield self.log_progress("❌ Concatenation failed"), "\n".join(self.summaries)
                    
                    # Release processing flag immediately so next cycle can start
                    self.processing = False
                else:
                    # Show waiting status with live file size update like main.py
                    if segments:
                        try:
                            # Get the current segment being recorded (highest index by filename)
                            current_segment = max(segments, key=lambda s: int(s.stem.split('_')[1]) if s.stem.split('_')[1].isdigit() else -1)
                            current_size = current_segment.stat().st_size
                            size_mb = current_size / (1024 * 1024)
                            
                            # FILE GROWTH DETECTION: Check if segment file stopped growing
                            if current_size == self.last_segment_size:
                                # File size hasn't changed
                                if self.segment_stall_check_time is None:
                                    # First time detecting no growth - start timer
                                    self.segment_stall_check_time = time.time()
                                elif time.time() - self.segment_stall_check_time >= 3:
                                    # File hasn't grown for 3+ seconds - RESTART FFmpeg
                                    stall_duration = int(time.time() - self.segment_stall_check_time)
                                    yield self.log_progress(f"🚨 STALL DETECTED: {current_segment.name} not growing for {stall_duration}s at {size_mb:.1f} MB"), "\n".join(self.summaries)
                                    
                                    # Prevent restart spam (max 1 restart per 30 seconds)
                                    if time.time() - self.last_restart_time < 30:
                                        yield self.log_progress(f"⚠️ Skipping restart (last restart was {int(time.time() - self.last_restart_time)}s ago)"), "\n".join(self.summaries)
                                    else:
                                        yield self.log_progress("🔄 Attempting to restart FFmpeg..."), "\n".join(self.summaries)
                                        
                                        if self.restart_ffmpeg(segment_duration, segments_dir):
                                            yield self.log_progress("✅ FFmpeg restarted successfully"), "\n".join(self.summaries)
                                        else:
                                            yield self.log_progress("❌ FFmpeg restart failed - stopping"), "\n".join(self.summaries)
                                            self.should_stop = True
                                            break
                            else:
                                # File is growing - reset stall detection
                                if self.segment_stall_check_time is not None:
                                    # Was stalled, now growing again
                                    yield self.log_progress(f"✅ Recording resumed: {current_segment.name} now at {size_mb:.1f} MB"), "\n".join(self.summaries)
                                self.segment_stall_check_time = None
                                self.last_segment_size = current_size
                            
                            # Show as "Finalizing" if we have valid max_index and it's the last segment needed
                            if max_index >= 0:
                                if self.last_end_index == -1:
                                    needed_index = num_segments - 1
                                else:
                                    needed_index = self.last_end_index + num_segments - overlap_segments
                                
                                if max_index >= needed_index:
                                    # We're ready to process, show finalizing status
                                    yield self.log_progress(f"🔄 Finalizing: {current_segment.name} ({size_mb:.1f} MB)"), "\n".join(self.summaries)
                                else:
                                    # Still waiting, show progress
                                    remaining = needed_index - max_index
                                    yield self.log_progress(f"⏳ Waiting for {remaining} more segments... | 📹 Recording: {current_segment.name} ({size_mb:.1f} MB)"), "\n".join(self.summaries)
                            else:
                                # No valid max_index yet, but show current recording
                                yield self.log_progress(f"⏳ Accumulating segments ({current_count}/{num_segments})... | 📹 Recording: {current_segment.name} ({size_mb:.1f} MB)"), "\n".join(self.summaries)
                        except (ValueError, OSError, ZeroDivisionError) as e:
                            # Fallback if we can't get segment info
                            yield self.log_progress(f"⏳ Waiting for segments... ({current_count} found)"), "\n".join(self.summaries)
                    else:
                        # No segments yet
                        yield self.log_progress(f"⏳ Waiting for first segment to start..."), "\n".join(self.summaries)
                
                # Check if background thread completed a summary
                if self.new_summary_available:
                    self.new_summary_available = False
                    # Force update of summaries panel
                    yield "\n".join(self.progress_log), "\n".join(self.summaries)
                
                time.sleep(5)
                
        except Exception as e:
            yield self.log_progress(f"❌ Error: {e}"), "\n".join(self.summaries)
        finally:
            # Stop recording if still running
            if self.recording_process and self.recording_process.poll() is None:
                yield self.log_progress("⏹️ Stopping recording..."), "\n".join(self.summaries)
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=5)
                except:
                    self.recording_process.kill()
                    self.recording_process.wait()
                yield self.log_progress("✅ Recording stopped"), "\n".join(self.summaries)
    
    def stop_recording(self):
        """Stop the recording process"""
        self.should_stop = True
        
        # Add to log
        self.log_progress("🛑 Stop requested...")
        
        # Terminate FFmpeg process
        if self.recording_process:
            try:
                self.log_progress("⏹️ Terminating FFmpeg recording...")
                self.recording_process.terminate()
                
                # Wait for process to end (with timeout)
                try:
                    self.recording_process.wait(timeout=5)
                    self.log_progress("✅ FFmpeg process terminated")
                except:
                    # If it doesn't terminate, force kill
                    self.log_progress("⚠️ Force killing FFmpeg process...")
                    self.recording_process.kill()
                    self.recording_process.wait()
                    self.log_progress("✅ FFmpeg process killed")
            except Exception as e:
                self.log_progress(f"❌ Error stopping FFmpeg: {e}")
        else:
            self.log_progress("⚠️ No recording process found")
        
        self.log_progress("✅ Stop complete")
        
        # Return current progress log
        return "\n".join(self.progress_log)

# Create Gradio interface
def create_interface():
    summarizer = LivestreamSummarizerGradio()
    
    with gr.Blocks(title="YouTube Livestream Summarizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎥 YouTube Livestream Summarizer
        ### AI-Powered Real-Time Livestream Analysis
        Records and summarizes YouTube livestreams using FFmpeg and Google's Gemini AI.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Configuration")
                
                youtube_url = gr.Textbox(
                    label="YouTube Livestream URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
                
                api_key = gr.Textbox(
                    label="Gemini API Key",
                    placeholder="Enter your Gemini API key",
                    type="password",
                    value=os.getenv('GEMINI_API_KEY', '')
                )
                
                model_name = gr.Dropdown(
                    label="Gemini Model",
                    choices=[
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash-lite"
                    ],
                    value="gemini-2.5-flash",
                    info="Choose the Gemini model for summarization"
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    video_duration = gr.Slider(
                        label="Video Clip Duration (seconds)",
                        minimum=60,
                        maximum=300,
                        value=120,
                        step=10,
                        info="Duration of each video clip to summarize"
                    )
                    
                    segment_duration = gr.Slider(
                        label="Segment Duration (seconds)",
                        minimum=5,
                        maximum=60,
                        value=10,
                        step=5,
                        info="Duration of each recording segment"
                    )
                    
                    overlap_segments = gr.Slider(
                        label="Overlap Segments",
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=1,
                        info="Number of overlapping segments between cycles"
                    )
                    
                    use_google_search = gr.Checkbox(
                        label="Enable Google Search Grounding",
                        value=False,
                        info="Allow Gemini to use Google Search for more context (may increase processing time)"
                    )
                
                with gr.Accordion("✏️ Custom Prompt", open=True):
                    prompt_text = gr.Textbox(
                        label="Gemini Prompt",
                        lines=6,
                        value="""liveposting, summary detail this stream for me in english
1. paragraph style, no bullets style
2. only provide liveposting nothing else, don't talk about you or something else outside the stream
3. don't mention timestamp of the video
4. simple english""",
                        info="Customize how Gemini should summarize the video"
                    )
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Recording & Summarizing", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Progress Log")
                progress_output = gr.Textbox(
                    label="Real-time Progress",
                    lines=20,
                    max_lines=20,
                    autoscroll=True,
                    show_copy_button=True
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Summaries")
                summary_output = gr.Textbox(
                    label="Generated Summaries",
                    lines=20,
                    max_lines=20,
                    autoscroll=True,
                    show_copy_button=True
                )
        
        gr.Markdown("""
        ---
        ### 📖 Instructions:
        1. **Get Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. **Paste YouTube URL**: Enter a live or streaming YouTube URL
        3. **Adjust Settings** (optional): Modify duration, segments, or prompt
        4. **Start**: Click "Start Recording & Summarizing"
        5. **Monitor**: Watch progress in left panel, summaries in right panel
        6. **Stop**: Click "Stop" when finished
        
        **Note**: Requires FFmpeg and yt-dlp installed. On Colab, these will be installed automatically.
        """)
        
        # Event handlers
        start_btn.click(
            fn=summarizer.run_summarizer,
            inputs=[youtube_url, api_key, video_duration, segment_duration, overlap_segments, model_name, prompt_text, use_google_search],
            outputs=[progress_output, summary_output]
        )
        
        stop_btn.click(
            fn=summarizer.stop_recording,
            outputs=[progress_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
