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

# Fix Windows console encoding for Unicode characters (Japanese, etc.)
if sys.platform == 'win32':
    # Set console to UTF-8 mode
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except:
        pass
    # Also set stdout/stderr encoding
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Configuration Constants
VIDEO_DURATION_SECONDS = 180  # Duration of video clips to send to Gemini (in seconds)
SEGMENT_DURATION = 30  # Duration of each video segment (in seconds)
NUM_SEGMENTS = VIDEO_DURATION_SECONDS // SEGMENT_DURATION  # Number of segments needed
OVERLAP_SEGMENTS = 0  # Number of overlapping segments between cycles
OVERLAP_SECONDS = OVERLAP_SEGMENTS * SEGMENT_DURATION  # Duration of overlap (calculated)

# Validate video duration to prevent Gemini rate limit issues
if VIDEO_DURATION_SECONDS < 60:
    print(f"❌ Error: VIDEO_DURATION_SECONDS must be at least 60 seconds to avoid Gemini API rate limits.")
    print(f"Current value: {VIDEO_DURATION_SECONDS} seconds")
    print(f"Please change VIDEO_DURATION_SECONDS in the script to 60 or more.")
    sys.exit(1)

# Retry Configuration
FFMPEG_MAX_RETRIES = 3  # Number of retries for FFmpeg operations (concatenation, compression)
FFMPEG_RETRY_DELAY = 120  # Seconds to wait between FFmpeg retries (2 minutes)
GEMINI_MAX_RETRIES = 3  # Number of retries for Gemini API calls
GEMINI_RETRY_DELAY = 30  # Seconds to wait between Gemini retries

# Stream Monitoring Configuration
STALL_TIMEOUT = 20 + SEGMENT_DURATION  # Seconds to wait before checking if stream has stalled (must be > SEGMENT_DURATION to avoid false positives)
MAX_STALL_WARNINGS = 3  # Number of consecutive stall warnings before considering stream ended (total: STALL_TIMEOUT * MAX_STALL_WARNINGS)

# Gemini Configuration
USE_GOOGLE_SEARCH = False  # Enable/disable Google Search grounding tool in Gemini
INCLUDE_PREVIOUS_SUMMARIES = 3  # Number of previous summaries to include as context (0 = none, 1+ = include that many for continuity)

# Load environment variables
load_dotenv()

# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Generate unique log filename with timestamp
from datetime import datetime
LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"run_{LOG_TIMESTAMP}.log"

# Configure logging with both console and file handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler for this run
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info(f"Log file: {LOG_FILE}")

# Suppress verbose HTTP logging from libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)

class LivestreamSummarizer:
    def __init__(self, hls_url: str, api_key: str, stream_name: str = None, custom_prompt: str = None, is_vod: bool = False):
        self.hls_url = hls_url
        self.api_key = api_key
        self.stream_name = stream_name or "stream"
        self.is_vod = is_vod  # Flag to indicate if this is VOD (uses yt-dlp pipe)
        
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
        self.yt_dlp_process = None  # For VOD mode
        self.recording_start_time = None
        self.program_start_time = time.time()  # Track program runtime for elapsed time display
        self.is_vod_mode = False  # Track if recording VOD (download completes quickly)
        self.ffmpeg_log_file = None
        
        # Overlap tracking
        self.last_end_index = -1  # Index of last segment used in previous cycle
        
        # Processing flag to prevent concurrent cycles
        self.processing = False
        
        # Consecutive validation failure counter (for detecting stuck/broken segments)
        self.consecutive_validation_failures = 0
        self.should_exit_due_to_failures = False  # Flag to signal exit from main loop
        
        # Storage for previous summaries (raw text only for context)
        self.summary_texts_only = []
        
        # Rate limiting for VOD (track last Gemini API call time)
        self.last_gemini_call_time = 0

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

        # Open log file for FFmpeg/yt-dlp stderr
        ffmpeg_log = Path(__file__).parent / "ffmpeg.log"
        self.ffmpeg_log_file = open(ffmpeg_log, 'wb')
        
        if self.is_vod:
            # VOD mode: yt-dlp streams raw data, FFmpeg handles ALL encoding/segmenting
            self.is_vod_mode = True  # Mark as VOD for special handling
            logger.info("VOD mode: yt-dlp raw stream → FFmpeg complete processing...")
            
            # yt-dlp: Download and stream raw container to stdout (NO transcoding)
            yt_dlp_cmd = [
                'yt-dlp',
                # Format: Best video+audio merged, prioritize MP4 container
                '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best',
                '--no-playlist',
                '--no-part',                          # No .part files
                '--newline',                          # Clean output
                '--no-warnings',
                # Performance: Multi-threaded download
                '--concurrent-fragments', '5',
                '--buffer-size', '16M',
                '--http-chunk-size', '10M',
                # Bypass bot detection - multiple clients
                '--extractor-args', 'youtube:player_client=android,android_music',
                '--user-agent', 'Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36',
                # Output complete container to stdout (yt-dlp merges if needed)
                '-o', '-',
                self.hls_url
            ]
            
            # FFmpeg: Receive complete stream, do ALL encoding/muxing/segmenting
            ffmpeg_cmd = [
                'ffmpeg',
                # Robust pipe input handling
                '-fflags', '+genpts+igndts+discardcorrupt',
                '-avoid_negative_ts', 'make_zero',
                '-thread_queue_size', '2048',             # Large buffer for smooth pipe read
                '-analyzeduration', '20M',                # Deep format analysis
                '-probesize', '20M',
                '-i', 'pipe:0',                           # Read merged stream from yt-dlp
                # Segmentation
                '-f', 'segment',
                '-segment_time', str(SEGMENT_DURATION),
                '-segment_format', 'mp4',
                '-segment_wrap', '0',
                '-reset_timestamps', '1',
                # Video: Complete re-encode for consistent output
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',                          # Balanced quality/speed
                '-tune', 'hq',
                '-rc', 'cbr',
                '-b:v', '800k',                           # 800kbps for better quality
                '-maxrate', '800k',
                '-bufsize', '1600k',
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',                    # Universal compatibility
                '-vf', 'scale=-2:720,fps=30',             # Force 720p30
                # Audio: Complete re-encode
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '48000',
                '-ac', '2',
                # MP4 optimizations
                '-movflags', '+faststart+frag_keyframe+empty_moov',
                '-brand', 'mp42',
                # Output
                str(self.segments_dir / 'segment_%03d.mp4')
            ]
            
            logger.info("VOD Pipeline: yt-dlp (download+merge) → pipe → FFmpeg (encode+segment)")
            logger.info(f"Output: 720p 800kbps segments in {self.segments_dir}")
            
            # Start yt-dlp: Downloads and merges video+audio, streams to stdout
            self.yt_dlp_process = subprocess.Popen(
                yt_dlp_cmd,
                stdout=subprocess.PIPE,                   # Pipe merged stream to FFmpeg
                stderr=self.ffmpeg_log_file,
                bufsize=16*1024*1024                      # 16MB buffer
            )
            
            # Start FFmpeg: Reads merged stream, encodes, segments
            self.recording_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self.yt_dlp_process.stdout,         # Read yt-dlp's merged output
                stdout=subprocess.DEVNULL,
                stderr=self.ffmpeg_log_file,
                bufsize=16*1024*1024
            )
            
            # CRITICAL: Keep pipe open! yt-dlp streams continuously to FFmpeg.
            # Closing stdout breaks the stream = audio-only or empty segments.
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] VOD recording started: streaming merged container to FFmpeg for processing")
            
        else:
            # Live stream mode: Direct FFmpeg recording from HLS URL (original behavior)
            logger.info("Live stream mode: Direct FFmpeg recording...")
            
            cmd = [
                'ffmpeg',
                # Network resilience options (before -i)
                '-reconnect', '1',                        # Enable reconnection on disconnect
                '-reconnect_streamed', '1',               # Reconnect even for streamed content
                '-reconnect_delay_max', '5',              # Max 5 seconds between reconnect attempts
                '-reconnect_on_network_error', '1',       # Reconnect on network errors
                '-reconnect_on_http_error', '4xx,5xx',    # Reconnect on HTTP errors
                '-rw_timeout', '10000000',                # 10 second read/write timeout (microseconds)
                '-timeout', '10000000',                   # 10 second connection timeout (microseconds)
                '-analyzeduration', '10M',                # Analyze up to 10MB for format detection
                '-probesize', '10M',                      # Probe up to 10MB for format detection
                '-fflags', '+genpts+discardcorrupt',      # Generate PTS, discard corrupt frames
                '-flags', 'low_delay',                    # Low latency mode
                '-strict', 'experimental',                # Allow experimental features
                '-i', self.hls_url,
                '-f', 'segment',
                '-segment_time', str(SEGMENT_DURATION),
                '-segment_wrap', '0',
                '-reset_timestamps', '1',
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-rc', 'cbr',
                '-b:v', '500k',
                '-maxrate', '500k',
                '-bufsize', '500k',
                '-vf', 'scale=-2:720,fps=30',
                '-c:a', 'aac',
                '-b:a', '64k',
                '-movflags', '+faststart',
                str(self.segments_dir / 'segment_%03d.mp4')
            ]
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting FFmpeg recording process with compression...")
            logger.info(f"Recording compressed segments to {self.segments_dir} ({SEGMENT_DURATION}s each)")
            
            self.recording_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self.ffmpeg_log_file)
            self.yt_dlp_process = None  # No yt-dlp for live streams
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Live stream recording started")
        
        logger.info("Compression settings: 720p H.264 @ 500k CBR video + 64k audio (optimized for speed)")
        self.recording_start_time = time.time()
        time.sleep(5)  # Wait for segments to start
        logger.info("Recording started successfully")
        logger.info(f"Output logged to: {ffmpeg_log}")

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
                    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Segment {segment_path.name} validated successfully")
                return True
            else:
                if log_on_failure:
                    logger.error(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Segment {segment_path.name} is invalid: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            if log_on_failure:
                logger.error(f"ffprobe timed out for {segment_path.name}")
            return False
        except Exception as e:
            if log_on_failure:
                logger.error(f"Error validating {segment_path.name}: {e}")
            return False

    def restart_recording(self):
        """Restart FFmpeg recording for livestreams (not VOD)."""
        if self.is_vod_mode:
            logger.warning("Cannot restart recording in VOD mode")
            return False
        
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Restarting FFmpeg to recover from network stall...")
        
        try:
            # Stop current recording
            if self.recording_process:
                logger.info("Stopping FFmpeg recording process...")
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=5)
                except:
                    self.recording_process.kill()
                    self.recording_process.wait()
                logger.info("FFmpeg stopped.")
            
            # Close old log file
            if self.ffmpeg_log_file:
                try:
                    self.ffmpeg_log_file.close()
                except:
                    pass
            
            # Wait for file handles to be released
            time.sleep(1)
            
            # Get current segment count and find the broken segment (the last one)
            segments = sorted(self.segments_dir.glob('segment_*.mp4'))
            restart_segment = 0
            
            if segments:
                try:
                    max_index = max(int(seg.stem.split('_')[1]) for seg in segments)
                    broken_segment = self.segments_dir / f"segment_{max_index:03d}.mp4"
                    
                    # Delete the broken segment (it was being written when FFmpeg was interrupted)
                    if broken_segment.exists():
                        broken_size = broken_segment.stat().st_size
                        broken_segment.unlink()
                        logger.info(f"🗑️ Deleted broken segment {broken_segment.name} ({broken_size / (1024*1024):.1f} MB)")
                    
                    # Restart from the same segment number (to replace the broken one)
                    restart_segment = max_index
                    logger.info(f"Restarting from segment {restart_segment:03d} (replacing broken segment)")
                except (ValueError, IndexError):
                    restart_segment = 0
                    logger.warning("Could not parse segment numbers, starting from 0")
            else:
                restart_segment = 0
                logger.info("No segments found, starting from 0")
            
            # Reopen log file
            ffmpeg_log = Path(__file__).parent / "ffmpeg.log"
            self.ffmpeg_log_file = open(ffmpeg_log, 'ab')  # Append mode
            
            # Restart FFmpeg (live stream mode only) with network resilience options
            cmd = [
                'ffmpeg',
                # Network resilience options (before -i)
                '-reconnect', '1',                        # Enable reconnection on disconnect
                '-reconnect_streamed', '1',               # Reconnect even for streamed content
                '-reconnect_delay_max', '5',              # Max 5 seconds between reconnect attempts
                '-reconnect_on_network_error', '1',       # Reconnect on network errors
                '-reconnect_on_http_error', '4xx,5xx',    # Reconnect on HTTP errors
                '-rw_timeout', '10000000',                # 10 second read/write timeout (microseconds)
                '-timeout', '10000000',                   # 10 second connection timeout (microseconds)
                '-analyzeduration', '10M',                # Analyze up to 10MB for format detection
                '-probesize', '10M',                      # Probe up to 10MB for format detection
                '-fflags', '+genpts+discardcorrupt',      # Generate PTS, discard corrupt frames
                '-flags', 'low_delay',                    # Low latency mode
                '-strict', 'experimental',                # Allow experimental features
                '-i', self.hls_url,
                '-f', 'segment',
                '-segment_time', str(SEGMENT_DURATION),
                '-segment_start_number', str(restart_segment),  # Replace the broken segment
                '-segment_wrap', '0',
                '-reset_timestamps', '1',
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-rc', 'cbr',
                '-b:v', '500k',
                '-maxrate', '500k',
                '-bufsize', '500k',
                '-vf', 'scale=-2:720,fps=30',
                '-c:a', 'aac',
                '-b:a', '64k',
                '-movflags', '+faststart',
                str(self.segments_dir / 'segment_%03d.mp4')
            ]
            
            self.recording_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self.ffmpeg_log_file)
            self.recording_start_time = time.time()
            time.sleep(3)  # Wait for FFmpeg to initialize
            
            logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ FFmpeg restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart FFmpeg: {e}")
            return False

    def stop_recording(self):
        """Stop the recording process."""
        # Stop yt-dlp process first (if VOD mode)
        if self.yt_dlp_process:
            logger.info("Stopping yt-dlp process...")
            try:
                self.yt_dlp_process.terminate()
                self.yt_dlp_process.wait(timeout=5)
            except:
                self.yt_dlp_process.kill()
            logger.info("yt-dlp stopped.")
        
        # Stop FFmpeg process
        if self.recording_process:
            logger.info("Stopping FFmpeg recording process...")
            self.recording_process.terminate()
            self.recording_process.wait()
            time.sleep(2)  # Wait for files to be fully written
            logger.info("Recording stopped.")
        else:
            logger.info("No recording process to stop.")
        
        # Close log file
        if self.ffmpeg_log_file:
            try:
                self.ffmpeg_log_file.close()
            except:
                pass

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

        # Check if recording is still active (FFmpeg running)
        recording_active = self.recording_process and self.recording_process.poll() is None
        
        if recording_active:
            # OPTIMIZATION: Instead of waiting for last segment to complete,
            # wait for NEXT segment to start (proves all cycle segments are complete)
            next_segment_index = end_index + 1
            next_segment_path = self.segments_dir / f"segment_{next_segment_index:03d}.mp4"
            
            logger.info(f"Waiting for next segment {next_segment_path.name} to start (ensures cycle segments are complete)...")
            wait_start = time.time()
            wait_timeout = SEGMENT_DURATION * 2  # Max 60 seconds wait
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
                    
                    # If FFmpeg is still running but segment is broken, trigger restart (live stream only)
                    if not self.is_vod_mode and self.recording_process and self.recording_process.poll() is None:
                        logger.warning("🚨 Segment validation failed during active recording - likely network stall")
                        logger.info("🔄 Attempting FFmpeg restart to recover...")
                        if self.restart_recording():
                            logger.info("✅ FFmpeg restarted after segment validation failure")
                            # Return False to skip this cycle, next cycle will use new segments
                            return False
                        else:
                            logger.error("❌ FFmpeg restart failed after segment validation failure")
                            return False
                    else:
                        # VOD mode or FFmpeg already stopped - just fail the cycle
                        return False
        else:
            # Recording finished (FFmpeg exited), all segments already complete
            logger.info(f"Recording complete, validating cycle segments {start_index} to {end_index}...")
            # Just validate the last segment to ensure it's complete
            last_segment_path = self.segments_dir / f"segment_{end_index:03d}.mp4"
            if last_segment_path.exists():
                if not self.validate_segment(last_segment_path, log_on_success=True, log_on_failure=True):
                    logger.error(f"Failed to validate {last_segment_path.name}")
                    return False
            else:
                logger.error(f"Missing final segment: {last_segment_path.name}")
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
            
            # Track consecutive validation failures
            self.consecutive_validation_failures += 1
            logger.warning(f"Consecutive validation failures: {self.consecutive_validation_failures}/{MAX_STALL_WARNINGS}")
            
            # If too many consecutive failures, try to restart or exit
            if self.consecutive_validation_failures >= MAX_STALL_WARNINGS:
                # Check if FFmpeg is still running
                if self.recording_process and self.recording_process.poll() is None:
                    # FFmpeg running but producing broken segments - try restart (live only)
                    if not self.is_vod_mode:
                        logger.warning(f"🚨 {MAX_STALL_WARNINGS} consecutive validation failures - attempting FFmpeg restart")
                        if self.restart_recording():
                            logger.info("✅ FFmpeg restarted after validation failures")
                            self.consecutive_validation_failures = 0  # Reset counter
                        else:
                            logger.error("❌ FFmpeg restart failed")
                else:
                    # FFmpeg has exited and segments are broken - signal to exit
                    logger.error(f"🛑 FFmpeg exited and {MAX_STALL_WARNINGS} consecutive validation failures")
                    logger.error("Stream has ended with corrupted final segments")
                    # Mark as needing exit by setting a flag
                    self.should_exit_due_to_failures = True
            
            return False
        
        # Reset failure counter on successful validation
        self.consecutive_validation_failures = 0
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
            concat_start_time = datetime.now()
            logger.info(f"[{concat_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting concatenation of {NUM_SEGMENTS} pre-compressed segments into {self.compressed_file} (attempt {attempt}/{max_retries})")
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
                    concat_end_time = datetime.now()
                    concat_duration = (concat_end_time - concat_start_time).total_seconds()
                    logger.info(f"[{concat_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Concatenation completed successfully in {concat_duration:.1f}s. Video file: {self.compressed_file} ({self.compressed_file.stat().st_size} bytes)")
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
            print("⏳ Waiting for file to become ACTIVE...", end='', flush=True)
            poll_count = 0
            while video_file.state.name == "PROCESSING":
                print(".", end='', flush=True)  # Show progress dots
                time.sleep(5)
                video_file = self.client.files.get(name=video_file.name)
                poll_count += 1
                if poll_count % 6 == 0:  # Log every 30 seconds (6 * 5s = 30s)
                    print(f" ({poll_count * 5}s)", end='', flush=True)
            print()  # New line after polling complete
            
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
            
            # Build prompt with previous summaries if requested
            final_prompt = self.custom_prompt
            
            if INCLUDE_PREVIOUS_SUMMARIES > 0 and len(self.summary_texts_only) > 0:
                # Validate the number
                total_summaries = len(self.summary_texts_only)
                
                if INCLUDE_PREVIOUS_SUMMARIES > total_summaries:
                    # Log warning but continue with available summaries
                    logger.warning(f"Requested {INCLUDE_PREVIOUS_SUMMARIES} previous summaries, but only {total_summaries} available. Using all {total_summaries}.")
                    include_count = total_summaries
                else:
                    include_count = INCLUDE_PREVIOUS_SUMMARIES
                
                # Get the requested number of previous summaries (most recent ones)
                previous_summaries = self.summary_texts_only[-include_count:]
                
                # Validate that we actually have summaries
                if not previous_summaries:
                    logger.warning("No previous summaries available, proceeding without context.")
                else:
                    # Build context section
                    context_section = "\n\n[PREVIOUS SUMMARIES FOR CONTEXT]\n"
                    context_section += "=" * 60 + "\n"
                    for i, prev_summary in enumerate(previous_summaries, 1):
                        # Ensure summary is not None or empty
                        if prev_summary and prev_summary.strip():
                            context_section += f"\nPrevious Summary #{i}:\n{prev_summary}\n"
                    context_section += "=" * 60 + "\n\n"
                    
                    # Insert context before the main prompt
                    final_prompt = context_section + self.custom_prompt
                    
                    logger.info(f"Including {len(previous_summaries)} previous summary/summaries for context")
            
            # Retry logic for transient Gemini API errors (500 INTERNAL, rate limits)
            max_retries = GEMINI_MAX_RETRIES
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            types.Part.from_text(text=final_prompt),
                            types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
                        ],
                        config=config
                    )
                    summary = response.text
                    break  # Success, exit retry loop
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    
                    # Check if it's a transient error (500, rate limit, etc.)
                    is_transient_error = any([
                        '500' in error_str,
                        'internal' in error_str,
                        'rate limit' in error_str,
                        'quota' in error_str,
                        '429' in error_str,
                        'resource exhausted' in error_str
                    ])
                    
                    if is_transient_error and attempt < max_retries - 1:
                        # Use GEMINI_RETRY_DELAY for rate limits and transient errors
                        logger.warning(f"Gemini API transient error (attempt {attempt + 1}/{max_retries}): {api_error}")
                        logger.warning(f"Retrying in {GEMINI_RETRY_DELAY}s...")
                        print(f"⚠️ Gemini API temporary error, retrying in {GEMINI_RETRY_DELAY}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(GEMINI_RETRY_DELAY)
                        continue
                    else:
                        # Not a transient error or final attempt, re-raise
                        raise
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
        
        # Store raw summary text for context feature
        self.summary_texts_only.append(summary_text)
        
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
            
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Starting summarization cycle ===")
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

    def process_remaining_segments(self, concatenate_all: bool = False):
        """Process any remaining segments when stream ends.
        
        Args:
            concatenate_all: If True, concatenate ALL remaining segments into one video
                           (used for livestream where remaining is usually small).
                           If False, process in cycles of NUM_SEGMENTS (used for VOD
                           where remaining can be large).
        
        Livestream: concatenate_all=True → All remaining in 1 video (typically few segments)
        VOD: concatenate_all=False → Process in cycles of 180s (can be many segments)
        """
        mode_desc = "single video" if concatenate_all else "cycles"
        logger.info(f"=== Processing remaining segments ({mode_desc} mode) ===")
        
        # Wait for any ongoing processing to complete
        while self.processing:
            logger.info("Waiting for current processing to complete...")
            time.sleep(2)
        
        # Get all remaining segments
        segments = sorted(self.segments_dir.glob('segment_*.mp4'))
        if not segments:
            logger.info("No segments found")
            return
        
        # Get segment indices
        try:
            max_available_index = max(int(seg.stem.split('_')[1]) for seg in segments)
        except (ValueError, IndexError):
            logger.warning("Could not parse segment indices")
            return
        
        # Determine start index
        if self.last_end_index == -1:
            start_index = 0
        else:
            start_index = max(0, self.last_end_index - OVERLAP_SEGMENTS + 1)
        
        # Check if there are any unprocessed segments
        if start_index > max_available_index:
            logger.info("All segments have been processed")
            return
        
        num_remaining = max_available_index - start_index + 1
        logger.info(f"Remaining segments: {start_index} to {max_available_index} ({num_remaining} segments)")
        
        # === CONCATENATE ALL MODE (Livestream) ===
        if concatenate_all:
            logger.info(f"Livestream mode: Concatenating all {num_remaining} remaining segments into one video")
            
            # Get all remaining segment files
            remaining_segments = []
            for i in range(start_index, max_available_index + 1):
                seg_path = self.segments_dir / f"segment_{i:03d}.mp4"
                if seg_path.exists():
                    remaining_segments.append(seg_path)
                else:
                    logger.warning(f"Missing segment: {seg_path.name}")
            
            if not remaining_segments:
                logger.info("No valid segments to process")
                return
            
            # Validate segments
            valid_segments = []
            for seg in remaining_segments:
                if self.validate_segment(seg, log_on_success=False, log_on_failure=True):
                    valid_segments.append(seg)
                else:
                    logger.warning(f"Skipping invalid segment: {seg.name}")
            
            if not valid_segments:
                logger.warning("No valid segments found")
                return
            
            total_duration = len(valid_segments) * SEGMENT_DURATION
            logger.info(f"Processing {len(valid_segments)} valid segments (~{total_duration}s total)")
            
            # Create concat file
            with open(self.concat_file, 'w') as f:
                for seg in valid_segments:
                    f.write(f"file '{seg}'\n")
            
            # Concatenate
            concat_start_time = datetime.now()
            logger.info(f"[{concat_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Concatenating all remaining segments...")
            
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(self.concat_file), '-c', 'copy',
                str(self.compressed_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    logger.error(f"Failed to concatenate: {result.stderr}")
                    return
                
                concat_end_time = datetime.now()
                logger.info(f"[{concat_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Concatenation completed")
                
            except Exception as e:
                logger.error(f"Error concatenating: {e}")
                return
            
            # Summarize with Gemini
            gemini_start_time = datetime.now()
            logger.info(f"[{gemini_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Sending final video to Gemini...")
            
            summary = self.summarize_with_gemini()
            
            if summary:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elapsed_seconds = int(time.time() - self.program_start_time)
                elapsed_str = f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}"
                
                logger.info(f"[{timestamp}] Final summary received (Elapsed: {elapsed_str})")
                print(f"\n--- Final Summary at {timestamp} (⏱️ {elapsed_str}) ---\n{summary}\n")
                summary_num = self.append_summary(summary, timestamp)
                print(f"✅ Final Summary #{summary_num} saved to summary-{self.stream_name}.txt")
            else:
                logger.error("Failed to generate final summary")
            
            # Cleanup
            if self.compressed_file.exists():
                try:
                    self.compressed_file.unlink()
                except:
                    pass
            
            logger.info("=== Remaining segments processing completed (single video) ===")
            return
        
        # === CYCLE MODE (VOD) ===
        cycle_count = 0
        
        while True:
            segments = sorted(self.segments_dir.glob('segment_*.mp4'))
            if not segments:
                logger.info("No segments found")
                break
            
            # Get the highest segment index available
            try:
                max_available_index = max(int(seg.stem.split('_')[1]) for seg in segments)
            except (ValueError, IndexError):
                logger.warning("Could not parse segment indices")
                break
            
            # Determine start index for this cycle
            if self.last_end_index == -1:
                start_index = 0
            else:
                start_index = max(0, self.last_end_index - OVERLAP_SEGMENTS + 1)
            
            end_index = start_index + NUM_SEGMENTS - 1
            
            # Check if we have enough segments for a full cycle
            if end_index <= max_available_index:
                logger.info(f"Processing cycle: segments {start_index} to {end_index} ({NUM_SEGMENTS} segments)")
            elif start_index <= max_available_index:
                # Partial cycle - process remaining segments
                end_index = max_available_index
                num_remaining = end_index - start_index + 1
                
                if num_remaining == 0:
                    logger.info("All segments have been processed")
                    break
                    
                logger.info(f"Processing final partial cycle: segments {start_index} to {end_index} ({num_remaining} segments)")
            else:
                logger.info("All segments have been processed")
                break
            
            # Get segment files for this cycle
            cycle_segments = []
            for i in range(start_index, end_index + 1):
                seg_path = self.segments_dir / f"segment_{i:03d}.mp4"
                if seg_path.exists():
                    cycle_segments.append(seg_path)
                else:
                    logger.warning(f"Missing segment: {seg_path.name}")
            
            if not cycle_segments:
                logger.info("No valid segments for this cycle")
                break
            
            # Validate segments
            valid_segments = []
            for seg in cycle_segments:
                if self.validate_segment(seg, log_on_success=False, log_on_failure=True):
                    valid_segments.append(seg)
                else:
                    logger.warning(f"Skipping invalid segment: {seg.name}")
            
            if not valid_segments:
                logger.warning("No valid segments in this cycle, skipping")
                self.last_end_index = end_index
                continue
            
            # Create concat file
            with open(self.concat_file, 'w') as f:
                for seg in valid_segments:
                    f.write(f"file '{seg}'\n")
            
            logger.info(f"Concat file created: {valid_segments[0].name} to {valid_segments[-1].name}")
            
            # Concatenate segments
            concat_start_time = datetime.now()
            logger.info(f"[{concat_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Concatenating {len(valid_segments)} segments...")
            
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(self.concat_file), '-c', 'copy',
                str(self.compressed_file)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"Failed to concatenate segments: {result.stderr}")
                    self.last_end_index = end_index
                    continue
                
                concat_end_time = datetime.now()
                concat_duration = (concat_end_time - concat_start_time).total_seconds()
                logger.info(f"[{concat_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Concatenation completed in {concat_duration:.1f}s")
                
            except subprocess.TimeoutExpired:
                logger.error("Timeout while concatenating segments")
                self.last_end_index = end_index
                continue
            except Exception as e:
                logger.error(f"Error concatenating segments: {e}")
                self.last_end_index = end_index
                continue
            
            # Update last_end_index
            self.last_end_index = end_index
            
            # Summarize with Gemini
            cycle_count += 1
            gemini_start_time = datetime.now()
            logger.info(f"[{gemini_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Sending to Gemini (remaining cycle #{cycle_count})...")
            
            summary = self.summarize_with_gemini()
            
            if summary:
                gemini_end_time = datetime.now()
                gemini_duration = (gemini_end_time - gemini_start_time).total_seconds()
                timestamp = gemini_end_time.strftime('%Y-%m-%d %H:%M:%S')
                
                elapsed_seconds = int(time.time() - self.program_start_time)
                elapsed_str = f"{elapsed_seconds // 3600:02d}:{(elapsed_seconds % 3600) // 60:02d}:{elapsed_seconds % 60:02d}"
                
                logger.info(f"[{timestamp}] Summary received (Gemini took {gemini_duration:.1f}s, Elapsed: {elapsed_str})")
                print(f"\n--- Summary at {timestamp} (⏱️ {elapsed_str}) ---\n{summary}\n")
                summary_num = self.append_summary(summary, timestamp)
                print(f"✅ Summary #{summary_num} saved to summary-{self.stream_name}.txt")
            else:
                logger.error("Failed to generate summary for this cycle")
            
            # Clean up compressed file
            if self.compressed_file.exists():
                try:
                    self.compressed_file.unlink()
                except:
                    pass
            
            # Clean up old segments
            self.cleanup_old_segments()
        
        logger.info("=== Remaining segments processing completed (cycles) ===")

    def _background_summarization(self):
        """Background thread for video processing and Gemini summarization.
        
        Note: Segments are already compressed during recording, so no separate
        compression step is needed. Concatenation outputs directly to compressed_file.
        """
        try:
            gemini_start_time = datetime.now()
            logger.info(f"[{gemini_start_time.strftime('%Y-%m-%d %H:%M:%S')}] Sending pre-compressed video to Gemini for summarization...")
            
            # Retry logic for Gemini summarization
            max_retries = GEMINI_MAX_RETRIES
            retry_delay = GEMINI_RETRY_DELAY
            summary = None
            
            for attempt in range(1, max_retries + 1):
                if attempt > 1:
                    logger.info(f"Retrying summarization (attempt {attempt}/{max_retries})...")
                
                summary = self.summarize_with_gemini()
                
                if summary:
                    gemini_end_time = datetime.now()
                    gemini_duration = (gemini_end_time - gemini_start_time).total_seconds()
                    timestamp = gemini_end_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Calculate elapsed time
                    elapsed_seconds = int(time.time() - self.program_start_time)
                    hours = elapsed_seconds // 3600
                    minutes = (elapsed_seconds % 3600) // 60
                    seconds = elapsed_seconds % 60
                    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    logger.info(f"[{timestamp}] Summary received (Gemini took {gemini_duration:.1f}s, Elapsed: {elapsed_str})")
                    print(f"\n--- Summary at {timestamp} (⏱️ {elapsed_str}) ---\n{summary}\n")
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
        no_growth_start = None
        last_max_index = -1
        stall_warning_count = 0  # Track consecutive stall warnings
        vod_complete_logged = False  # Track if VOD completion message already shown
        ffmpeg_exited = False  # Track if FFmpeg has exited (stop recording status display)
        
        try:
            while True:
                schedule.run_pending()  # Keep for any other scheduled tasks
                
                # Cache FFmpeg poll result to avoid redundant syscalls
                ffmpeg_has_exited = self.recording_process and self.recording_process.poll() is not None
                
                # Check if we should exit due to consecutive validation failures
                if self.should_exit_due_to_failures:
                    print()  # New line after recording status
                    logger.error("🛑 Exiting due to consecutive segment validation failures")
                    logger.info("Processing remaining valid segments before exit...")
                    # Livestream: concatenate all remaining into 1 video
                    # VOD: process in cycles
                    self.process_remaining_segments(concatenate_all=not self.is_vod_mode)
                    logger.info("All processing completed. Exiting.")
                    break
                
                # Check if FFmpeg process has exited (stream ended)
                if ffmpeg_has_exited:
                    print()  # New line after the recording status
                    exit_code = self.recording_process.returncode
                    runtime = time.time() - self.recording_start_time if self.recording_start_time else 0
                    
                    # Close log file and read its contents
                    if self.ffmpeg_log_file:
                        try:
                            self.ffmpeg_log_file.close()
                            self.ffmpeg_log_file = None
                        except:
                            pass
                    
                    # Read FFmpeg log to check for errors
                    stderr_text = ""
                    ffmpeg_log = Path(__file__).parent / "ffmpeg.log"
                    if ffmpeg_log.exists():
                        try:
                            with open(ffmpeg_log, 'r', encoding='utf-8', errors='ignore') as f:
                                stderr_text = f.read()
                        except:
                            pass
                    
                    # Determine if this was a clean exit or an error
                    is_error = False
                    error_indicators = [
                        "Connection refused",
                        "Server returned 404",
                        "Server returned 403",
                        "Invalid data found",
                        "Immediate exit requested",
                        "Conversion failed",
                        "moov atom not found",
                        "Protocol not found",
                        "No such file or directory"
                    ]
                    
                    # Check if FFmpeg exited too quickly AND no segments were created (likely an error)
                    segments_created = len(list(self.segments_dir.glob('segment_*.mp4')))
                    if runtime < 30 and segments_created == 0:
                        is_error = True
                        logger.warning(f"FFmpeg exited after only {runtime:.1f} seconds with no segments - likely an error")
                    
                    # Check stderr for error messages (but ignore common warnings)
                    if stderr_text:
                        for indicator in error_indicators:
                            if indicator.lower() in stderr_text.lower():
                                is_error = True
                                logger.error(f"FFmpeg error detected: {indicator}")
                                break
                    
                    if is_error:
                        logger.error(f"FFmpeg process exited with code {exit_code} due to an error")
                        if stderr_text:
                            # Show last 1000 characters of stderr
                            logger.error(f"FFmpeg log (last 1000 chars):\n{stderr_text[-1000:]}")
                            print(f"\n❌ FFmpeg Error:\n{stderr_text[-1000:]}\n")
                        logger.error("Recording failed. Please check the stream URL and try again.")
                        logger.info(f"Full FFmpeg log available at: {ffmpeg_log}")
                        break
                    else:
                        if not ffmpeg_exited:  # Only process once
                            print()  # Clear the recording status line
                            ffmpeg_exited = True  # Stop recording status display
                        if self.is_vod_mode:
                            # VOD: Download complete, but continue cycling through segments normally
                            if not vod_complete_logged:
                                logger.info(f"FFmpeg process exited cleanly with code {exit_code} after {runtime:.1f} seconds")
                                logger.info("VOD download complete. Continuing to process segments in cycles...")
                                logger.info("Program will continue until all segments are summarized.")
                                vod_complete_logged = True  # Only log once
                            # Don't break - continue the loop to process remaining cycles
                        else:
                            logger.info(f"FFmpeg process exited cleanly with code {exit_code} after {runtime:.1f} seconds")
                            # Live stream: Natural end, process remaining segments
                            logger.info("Live stream ended naturally, processing remaining segments...")
                            
                            # Wait a moment for any final segments to be written
                            time.sleep(3)
                            
                            # Process any remaining segments (concatenate all for livestream)
                            self.process_remaining_segments(concatenate_all=True)
                            
                            logger.info("All processing completed. Exiting.")
                            break
                
                # Check and log segment accumulation progress
                segments = list(self.segments_dir.glob('segment_*.mp4'))
                current_count = len(segments)
                
                # Calculate required max index for next cycle
                current_max_index = -1
                if segments:
                    try:
                        current_max_index = max(int(seg.stem.split('_')[1]) for seg in segments if seg.stem.split('_')[1].isdigit())
                        
                        # Detect stalled recording (no new segments for STALL_TIMEOUT seconds)
                        if current_max_index == last_max_index:
                            if no_growth_start is None:
                                no_growth_start = time.time()
                            elif time.time() - no_growth_start > STALL_TIMEOUT:
                                print()  # New line after the recording status
                                # Check if FFmpeg is still running (use cached result)
                                if not ffmpeg_has_exited:
                                    # FFmpeg is still running but segments stopped - network stall
                                    stall_warning_count += 1
                                    logger.warning(f"No new segments for {STALL_TIMEOUT}s but FFmpeg still running - likely network stall (warning {stall_warning_count}/{MAX_STALL_WARNINGS})")
                                    
                                    if stall_warning_count >= MAX_STALL_WARNINGS:
                                        # After MAX_STALL_WARNINGS restart attempts, consider stream ended
                                        logger.warning(f"Stream stalled for {STALL_TIMEOUT * MAX_STALL_WARNINGS}s total ({MAX_STALL_WARNINGS} restart attempts failed). Considering stream ended.")
                                        logger.info("Processing remaining segments...")
                                        self.stop_recording()
                                        # Livestream stall: concatenate all remaining into 1 video
                                        self.process_remaining_segments(concatenate_all=True)
                                        logger.info("All processing completed. Exiting.")
                                        break
                                    else:
                                        # Attempt to restart FFmpeg to recover from network stall
                                        logger.info(f"🔄 Attempting FFmpeg restart (attempt {stall_warning_count}/{MAX_STALL_WARNINGS})...")
                                        if self.restart_recording():
                                            logger.info("✅ FFmpeg restarted, continuing from current segment")
                                            no_growth_start = time.time()  # Reset timer after restart
                                            # Keep stall_warning_count to track restart attempts
                                        else:
                                            logger.error("❌ FFmpeg restart failed")
                                            logger.warning("Continuing to monitor in case connection recovers...")
                                            no_growth_start = time.time()  # Reset timer to give more time
                                else:
                                    # FFmpeg has exited
                                    if self.is_vod_mode:
                                        # VOD: FFmpeg exited is normal, continue cycling through segments
                                        # Don't trigger remaining segments processing - let VOD completion check handle it
                                        logger.info(f"VOD: No new segments for {STALL_TIMEOUT}s (download complete), continuing to process existing segments...")
                                        no_growth_start = None  # Reset to avoid re-triggering
                                    else:
                                        # Live stream: FFmpeg exited + no new segments = stream truly ended
                                        logger.warning(f"No new segments detected for {STALL_TIMEOUT} seconds and FFmpeg has exited")
                                        logger.info("Processing remaining segments...")
                                        # Livestream: concatenate all remaining into 1 video
                                        self.process_remaining_segments(concatenate_all=True)
                                        logger.info("All processing completed. Exiting.")
                                        break
                        else:
                            no_growth_start = None
                            last_max_index = current_max_index
                            stall_warning_count = 0  # Reset counter when new segments arrive
                        
                        if self.last_end_index == -1:
                            required_max_index = NUM_SEGMENTS - 1
                        else:
                            required_max_index = self.last_end_index + NUM_SEGMENTS - OVERLAP_SEGMENTS
                        
                        if current_max_index >= required_max_index and not self.processing:
                            cycle_count += 1
                            logger.info(f"Starting summarization cycle #{cycle_count}")
                            self.process_and_summarize()
                    except ValueError:
                        logger.warning("Could not parse segment indices")
                
                # VOD mode: Check if download finished AND all segments processed
                if self.is_vod_mode and ffmpeg_has_exited:
                    # Download complete, check if all segments summarized
                    if not self.processing:  # No active summarization
                        segments_list = sorted(self.segments_dir.glob('segment_*.mp4'))
                        if segments_list:
                            try:
                                max_segment_index = max(int(seg.stem.split('_')[1]) for seg in segments_list if seg.stem.split('_')[1].isdigit())
                                # Check if we've processed all segments
                                if self.last_end_index >= max_segment_index:
                                    print()  # New line after recording status
                                    logger.info(f"VOD complete: All {max_segment_index + 1} segments have been summarized.")
                                    logger.info("All processing completed. Exiting.")
                                    break
                            except (ValueError, IndexError):
                                pass
                
                # Log segment progress only when count changes (and only if FFmpeg still running)
                if not ffmpeg_exited and current_count != last_segment_count and current_count < NUM_SEGMENTS:
                    logger.info(f"Segments accumulated: {current_count}/{NUM_SEGMENTS} ({NUM_SEGMENTS - current_count} remaining)")
                    last_segment_count = current_count
                
                # Update current recording file size every 5 seconds (only if FFmpeg is still running)
                if not ffmpeg_exited:
                    current_time = time.time()
                    if current_time - last_size_update >= 5:
                        # Calculate elapsed time
                        elapsed_seconds = int(current_time - self.program_start_time)
                        hours = elapsed_seconds // 3600
                        minutes = (elapsed_seconds % 3600) // 60
                        seconds = elapsed_seconds % 60
                        elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        if segments:
                            # Get the current segment being recorded (highest index)
                            try:
                                current_segment = max(segments, key=lambda s: int(s.stem.split('_')[1]))
                                new_size = current_segment.stat().st_size
                                if new_size != current_recording_size:
                                    current_recording_size = new_size
                                    # Update the same line with current file size and elapsed time
                                    size_mb = current_recording_size / (1024 * 1024)
                                    print(f"\r📹 Recording: {current_segment.name} ({size_mb:.1f} MB) | ⏱️ {elapsed_str}", end='', flush=True)
                            except (ValueError, OSError):
                                pass
                        else:
                            # Show elapsed time even if no segments yet
                            print(f"\r⏱️ Elapsed: {elapsed_str}", end='', flush=True)
                        
                        last_size_update = current_time
                
                # Slow down loop when FFmpeg has exited to avoid interfering with background processing
                if ffmpeg_exited and self.processing:
                    time.sleep(5)  # Slower polling when processing in background
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print()  # New line after the recording status
            logger.info("Stopping...")
            self.stop_recording()
            logger.info("Processing any remaining segments...")
            # Livestream: concatenate all remaining into 1 video
            # VOD: process in cycles
            self.process_remaining_segments(concatenate_all=not self.is_vod_mode)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python main.py <youtube_url_or_hls_url> [prompt]")
        print("  - If prompt is empty, uses default prompt")
        print("  - If prompt is provided, uses it for Gemini summarization")
        sys.exit(1)

    url = sys.argv[1]
    custom_prompt = sys.argv[2] if len(sys.argv) == 3 else None
    stream_name = None  # Will be auto-extracted from YouTube title
    
    # Check if it's a YouTube URL
    is_youtube = 'youtube.com' in url or 'youtu.be' in url
    is_vod = False
    
    if is_youtube:
        print("Detected YouTube URL, checking if live or VOD...")
        
        # Check if it's a livestream or VOD
        try:
            check_result = subprocess.run(
                ['yt-dlp', '--print', '%(is_live)s',
                 '--extractor-args', 'youtube:player_client=android,android_music',
                 '--user-agent', 'Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36',
                 url],
                capture_output=True,
                text=True,
                timeout=30
            )
            if check_result.returncode == 0:
                is_live_str = check_result.stdout.strip().lower()
                is_vod = is_live_str in ['false', 'none', '']
                if is_vod:
                    print("✓ Detected VOD (Video On Demand)")
                else:
                    print("✓ Detected Live Stream")
        except Exception as e:
            print(f"⚠️ Could not determine if live/VOD, assuming live: {e}")
            is_vod = False
        
        # Extract stream name from YouTube URL if not provided
        if not stream_name:
            try:
                # Get video title to use as stream name
                # Use --encoding utf-8 to properly handle Japanese/Unicode titles on Windows
                title_result = subprocess.run(
                    ['yt-dlp', '--encoding', 'utf-8', '--get-title', url],
                    capture_output=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=30
                )
                if title_result.returncode == 0 and title_result.stdout.strip():
                    # Sanitize title for filename (allow Unicode characters including Japanese)
                    import re
                    raw_title = title_result.stdout.strip()
                    # Remove only characters that are problematic for filenames on Windows
                    # Keep Unicode letters (Japanese, etc.), numbers, spaces, hyphens, underscores
                    stream_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', raw_title)
                    # Replace multiple spaces/hyphens with single hyphen, but keep Japanese chars
                    stream_name = re.sub(r'[\s]+', '-', stream_name)  # Spaces to hyphens
                    stream_name = re.sub(r'-+', '-', stream_name)  # Multiple hyphens to single
                    stream_name = stream_name.strip('-')[:80]  # Increased limit for Japanese titles
                    print(f"Stream name: {stream_name}")
            except Exception as e:
                print(f"Could not get stream name: {e}")
                stream_name = "stream"
        
        # For VOD, use YouTube URL directly (yt-dlp will handle it)
        if is_vod:
            hls_url = url  # Pass YouTube URL, start_recording will use yt-dlp pipe
            print("Using yt-dlp pipe method for VOD (no URL expiration issues)")
        else:
            # For live streams, extract HLS URL (original behavior)
            print("Extracting HLS stream URL for live stream...")
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
    summarizer = LivestreamSummarizer(hls_url, api_key, stream_name, custom_prompt, is_vod=is_vod)
    summarizer.run()

if __name__ == "__main__":
    main()