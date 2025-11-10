# YouTube Livestream Summarizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leoparker157/youtube-livestream-summarizer/blob/main/YouTube_Livestream_Summarizer.ipynb)

This Python program automatically records and summarizes YouTube livestreams in real time using FFmpeg and Google's Gemini API.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Run the app instantly in your browser - no installation required!

1. Click the "Open In Colab" badge above
2. Follow the notebook instructions
3. Start summarizing livestreams immediately

### Option 2: Local Installation
Run on your local machine for full control.

## Features

- 🚀 Real-time livestream recording with optimized compression
- 🤖 AI-powered summarization using Gemini 2.0 Flash
- ⚡ Speed-optimized for continuous processing
- 📊 Customizable video duration and segments
- 🌐 Web interface with Gradio
- 📝 Live progress monitoring and summary display
- 🔄 Continuous background summarization

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

### Google Colab

No installation required! The Colab notebook automatically installs all dependencies.

## Usage

### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python gradio_app.py
```

The interface provides:
- **Configuration Panel**: Set YouTube URL, API key, and advanced settings
- **Progress Log**: Real-time monitoring of recording and processing
- **Summaries Panel**: Clean display of AI-generated summaries
- **Custom Prompts**: Full control over how Gemini summarizes content

### Command Line Interface

Run the script with a YouTube URL or HLS URL:

```bash
python main.py <youtube_url>
# or
python main.py <hls_url>
```

The script will automatically detect YouTube URLs and extract the HLS stream using yt-dlp. For direct HLS URLs, it will use them as-is.

## Gradio Web Interface

The web interface provides an intuitive way to configure and monitor livestream summarization:

### Features
- **Real-time Progress Monitoring**: Live updates on recording and processing status
- **Dual-Panel Display**: Separate progress log and summaries panels
- **Customizable Settings**: Adjust video duration, segments, and overlap
- **Prompt Editor**: Full control over Gemini summarization prompts
- **One-Click Operation**: Start/stop with simple button controls

### Configuration Examples

#### Fast Processing (Quick Summaries)
- Video Clip Duration: 60 seconds
- Segment Duration: 10 seconds
- Overlap: 0

#### Standard Processing (Balanced)
- Video Clip Duration: 120 seconds
- Segment Duration: 10 seconds
- Overlap: 0

#### Deep Analysis (Detailed Summaries)
- Video Clip Duration: 180-300 seconds
- Segment Duration: 15 seconds
- Overlap: 1-2

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

### "Failed to extract HLS URL"
- Ensure the YouTube URL is correct and the stream is live
- Update yt-dlp: `pip install -U yt-dlp`
- Try using a direct HLS URL instead

### "Gemini API Error"
- Verify your API key is valid and has quota remaining
- Check rate limits in Google AI Studio
- Try reducing video clip duration

### "FFmpeg Error"
- Ensure FFmpeg is installed and accessible
- Restart your runtime/session
- Check if the stream URL is still valid

### Colab-Specific Issues
- Use GPU runtime for faster processing
- Monitor disk usage for long sessions
- Session may timeout - save important summaries

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Links
- **Google AI Studio**: [Get API Key](https://aistudio.google.com/app/apikey)
- **Gemini API Docs**: [Google Generative AI](https://ai.google.dev/)
- **FFmpeg Documentation**: [ffmpeg.org](https://ffmpeg.org/documentation.html)