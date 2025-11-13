# YouTube Livestream Summarizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leoparker157/youtube-livestream-summarizer/blob/main/YouTube_Livestream_Summarizer.ipynb)

This Python program automatically records and summarizes YouTube livestreams in real time using FFmpeg and Google's Gemini API.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Run the app instantly in your browser - no installation required!

1. Click the "Open In Colab" badge above
2. Follow the notebook instructions
3. Start summarizing livestreams immediately

**Configuration**: Adjust video duration (60-300s), segments (10-30s), and overlap (0-2) for different processing speeds.

### Option 2: Local Installation
Run on your local machine for full control.

## Features

-Real-time summarization of livestreams using Gemini
-Allow customizable encoding, compression, and prompts


## Configuration

You can customize the behavior by modifying constants in `main.py`:

```python
# Video Processing
VIDEO_DURATION_SECONDS = 120  # Duration of video clips to send to Gemini (in seconds)
SEGMENT_DURATION = 60  # Duration of each video segment (minimum 60s to avoid rate limits)
OVERLAP_SEGMENTS = 0  # Number of overlapping segments between cycles

# Gemini Features
USE_GOOGLE_SEARCH = False  # Enable/disable Google Search grounding tool
INCLUDE_PREVIOUS_SUMMARIES = 0  # Number of previous summaries to include as context
                                # (0 = none, 1+ = include that many for continuity)
```

The program will automatically adjust the number of segments and processing accordingly.

**Context Feature**: Set `INCLUDE_PREVIOUS_SUMMARIES` to include previous summaries as context for better continuity. For example:
- `INCLUDE_PREVIOUS_SUMMARIES = 0` - Each summary is independent (default)
- `INCLUDE_PREVIOUS_SUMMARIES = 2` - Each summary includes the previous 2 summaries as context
- `INCLUDE_PREVIOUS_SUMMARIES = 5` - Each summary includes the previous 5 summaries for maximum continuity

This helps Gemini understand ongoing narratives and avoid repeating information.

### Configuration Examples

#### Fast Processing (Quick Summaries)
- Video Clip Duration: 60 seconds
- Segment Duration: 10 seconds
- Overlap: 0 (no overlap between segments)

#### Standard Processing (Balanced)
- Video Clip Duration: 120 seconds
- Segment Duration: 10 seconds
- Overlap: 0 (no overlap between segments)

#### Deep Analysis (Detailed Summaries)
- Video Clip Duration: 180-300 seconds
- Segment Duration: 15 seconds
- Overlap: 1-2 (segments overlap by 15-30 seconds for better context)

**Overlap**: Number of overlapping segments between cycles. Higher overlap provides better context continuity but increases processing time.

## Prerequisites

- Python 3.9+
- FFmpeg installed and on PATH
- yt-dlp installed (for YouTube URL support): `pip install yt-dlp`
- Google Gemini API key

## Installation

### Local Installation

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

### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python gradio_app.py
```

### Command Line Interface

Run the script with a YouTube URL and optional custom prompt:

```bash
# Use default prompt
python main.py <youtube_url>

# Use custom prompt
python main.py <youtube_url> "Your custom prompt here"
```

**Examples:**

```bash
# Default summarization
python main.py https://youtube.com/watch?v=...

# Custom gaming stream prompt
python main.py https://youtube.com/watch?v=... "Summarize the key gaming moments and funny reactions"

# Custom news stream prompt
python main.py https://youtube.com/watch?v=... "Extract main headlines and controversies discussed"

# Direct HLS URL with custom prompt
python main.py <hls_url> "Create a detailed technical summary"
```

**Features:**
- Automatically extracts HLS stream from YouTube URLs using yt-dlp
- Stream name auto-extracted from YouTube video title
- Custom prompts override the default summarization style
- Summaries saved to `summary-[video-title].txt`

## API Setup

### Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Or enter it directly in the Gradio interface

**Note**: Free tier has rate limits. Consider upgrading for heavy usage.

## Troubleshooting

### "Gemini API Error"
- Verify your API key is valid and has quota remaining
- Check rate limits in Google AI Studio
- Try reducing video clip duration

### Local Installation
- Ensure FFmpeg is installed and accessible
- Install yt-dlp: `pip install yt-dlp`

## License
MIT License - see LICENSE file for details
## Links
- **Google AI Studio**: [Get API Key](https://aistudio.google.com/app/apikey)
- **Gemini API Docs**: [Google Generative AI](https://ai.google.dev/)
- **FFmpeg Documentation**: [ffmpeg.org](https://ffmpeg.org/documentation.html)