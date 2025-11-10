# YouTube Livestream Summarizer

This Python program automatically records and summarizes YouTube livestreams in real time using FFmpeg and Google's Gemini API.

## Configuration

You can adjust the video duration sent to Gemini by modifying the `VIDEO_DURATION_SECONDS` constant in `main.py`:

```python
VIDEO_DURATION_SECONDS = 600  # 10 minutes (default)
# Or
VIDEO_DURATION_SECONDS = 300  # 5 minutes
```

The program will automatically adjust the number of segments and processing accordingly.

## Prerequisites

- Python 3.9+
- FFmpeg installed and on PATH
- yt-dlp installed (for YouTube URL support): `pip install yt-dlp`
- Google Gemini API key

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install yt-dlp for YouTube URL support:
   ```bash
   pip install yt-dlp
   ```

3. Set up your Gemini API key in a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the script with a YouTube URL or HLS URL:

```bash
python main.py <youtube_url>
# or
python main.py <hls_url>
```

The script will automatically detect YouTube URLs and extract the HLS stream using yt-dlp. For direct HLS URLs, it will use them as-is.

## Requirements

- ffmpeg
- google-genai
- schedule
- python-dotenv