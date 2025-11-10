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
        self.processing = False
        self.should_stop = False
        self.last_end_index = -1
        self.progress_log = []
        self.summaries = []
        
    def log_progress(self, message):
        """Add message to progress log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.progress_log.append(log_entry)
        logger.info(message)
        return "\n".join(self.progress_log)
    
    def add_summary(self, summary_text):
        """Add summary to summaries list"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        summary_entry = f"📝 Summary at {timestamp}\n{'='*60}\n{summary_text}\n\n"
        self.summaries.append(summary_entry)
        return "\n".join(self.summaries)
    
    def start_recording(self, hls_url, segment_duration, segments_dir):
        """Start FFmpeg recording"""
        cmd = [
            'ffmpeg',
            '-i', hls_url,
            '-f', 'segment',
            '-segment_time', str(segment_duration),
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
            str(segments_dir / 'segment_%03d.mp4')
        ]
        self.recording_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    
    def validate_segment(self, segment_path):
        """Validate segment file"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', str(segment_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def concatenate_segments(self, segments_dir, concat_file, compressed_file, num_segments):
        """Concatenate segments"""
        segments = sorted(segments_dir.glob('segment_*.mp4'))
        
        if len(segments) < num_segments:
            return False
        
        # Get latest segments
        latest_segments = segments[-num_segments:]
        
        # Create concat file
        with open(concat_file, 'w') as f:
            for seg in latest_segments:
                f.write(f"file '{seg}'\n")
        
        # Concatenate
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file), '-c', 'copy', str(compressed_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except:
            return False
    
    def summarize_with_gemini(self, compressed_file, api_key, prompt_text, use_google_search=False):
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
                self.log_progress("🔍 Generating summary with Google Search enabled...")
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(tools=[grounding_tool])
            else:
                self.log_progress("🤖 Generating summary...")
                config = types.GenerateContentConfig()
            
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
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
    
    def cleanup_old_segments(self, segments_dir, keep_count):
        """Clean up old segments"""
        segments = sorted(segments_dir.glob('segment_*.mp4'))
        if len(segments) > keep_count:
            for seg in segments[:-keep_count]:
                try:
                    seg.unlink()
                except:
                    pass
    
    def run_summarizer(self, youtube_url, api_key, video_duration, segment_duration, 
                       overlap_segments, prompt_text, use_google_search, progress=gr.Progress()):
        """Main summarization loop"""
        self.should_stop = False
        self.progress_log = []
        self.summaries = []
        
        # Setup directories
        script_dir = Path.cwd()
        segments_dir = script_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        concat_file = segments_dir / "concat.txt"
        compressed_file = script_dir / "compressed.mp4"
        
        num_segments = video_duration // segment_duration
        
        yield self.log_progress(f"⚙️ Configuration: {video_duration}s clips, {segment_duration}s segments, {num_segments} segments per cycle"), ""
        
        # Extract HLS URL
        yield self.log_progress("🔍 Extracting HLS URL from YouTube..."), ""
        
        try:
            result = subprocess.run(
                ['yt-dlp', '-f', 'best', '-g', youtube_url],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                yield self.log_progress("❌ Failed to extract HLS URL"), ""
                return
            hls_url = result.stdout.strip()
            yield self.log_progress(f"✅ HLS URL extracted"), ""
        except Exception as e:
            yield self.log_progress(f"❌ Error: {e}"), ""
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
                # Check segments
                segments = list(segments_dir.glob('segment_*.mp4'))
                current_count = len(segments)
                
                if current_count >= num_segments and not self.processing:
                    cycle_count += 1
                    self.processing = True
                    
                    yield self.log_progress(f"📊 Cycle #{cycle_count}: Processing {num_segments} segments..."), "\n".join(self.summaries)
                    
                    # Get segment details
                    segment_files = sorted(segments_dir.glob('segment_*.mp4'))[-num_segments:]
                    total_size = sum(seg.stat().st_size for seg in segment_files) / (1024 * 1024)
                    yield self.log_progress(f"📁 Using segments: {segment_files[0].name} to {segment_files[-1].name} ({total_size:.1f} MB)"), "\n".join(self.summaries)
                    
                    # Concatenate
                    yield self.log_progress("🔗 Concatenating segments..."), "\n".join(self.summaries)
                    if self.concatenate_segments(segments_dir, concat_file, compressed_file, num_segments):
                        final_size = compressed_file.stat().st_size / (1024 * 1024)
                        yield self.log_progress(f"✅ Concatenation complete ({final_size:.1f} MB)"), "\n".join(self.summaries)
                        
                        # Summarize
                        yield self.log_progress("🤖 Sending to Gemini AI..."), "\n".join(self.summaries)
                        summary = self.summarize_with_gemini(compressed_file, api_key, prompt_text, use_google_search)
                        
                        if summary:
                            yield self.log_progress(f"✅ Summary #{cycle_count} received"), self.add_summary(summary)
                        else:
                            yield self.log_progress("❌ Summary generation failed"), "\n".join(self.summaries)
                        
                        # Cleanup
                        yield self.log_progress("🧹 Cleaning up temporary files..."), "\n".join(self.summaries)
                        if compressed_file.exists():
                            compressed_file.unlink()
                        self.cleanup_old_segments(segments_dir, num_segments + 5)
                        yield self.log_progress("✅ Cleanup complete"), "\n".join(self.summaries)
                    else:
                        yield self.log_progress("❌ Concatenation failed"), "\n".join(self.summaries)
                    
                    self.processing = False
                else:
                    remaining = num_segments - current_count
                    if remaining > 0:
                        yield self.log_progress(f"⏳ Waiting for segments... ({current_count}/{num_segments})"), "\n".join(self.summaries)
                
                time.sleep(5)
                
        except Exception as e:
            yield self.log_progress(f"❌ Error: {e}"), "\n".join(self.summaries)
        finally:
            # Stop recording
            if self.recording_process:
                self.recording_process.terminate()
                self.recording_process.wait()
            yield self.log_progress("⏹️ Recording stopped"), "\n".join(self.summaries)
    
    def stop_recording(self):
        """Stop the recording process"""
        self.should_stop = True
        if self.recording_process:
            self.recording_process.terminate()
        return "Stopping..."

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
            inputs=[youtube_url, api_key, video_duration, segment_duration, overlap_segments, prompt_text, use_google_search],
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
