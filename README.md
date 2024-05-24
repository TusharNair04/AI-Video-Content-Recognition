# AI Video Content Recognition

This repository contains a script to generate a summary of a video by analyzing frames using a pre-trained ResNet-50 model and summarizing the frame descriptions using the LLaMA 3 API.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)

## Overview

The AI Video Content Recognition processes a video to extract frames at regular intervals, analyzes these frames using a pre-trained ResNet-50 model to generate descriptions, and uses the LLaMA 3 API to create a comprehensive summary of the video content.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/TusharNair04/AI-Video-Content-Recognition.git
   cd AI-Video-Content-Recognition
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Update the configuration file:**

   - Open `config/config.yaml` and update the `llama_api_key` and `video_path` with your API key and the path to your video file.

   Example `config/config.yaml`:

   ```yaml
   llama_api_key: "YOUR_LLAMA_API_KEY"
   video_path: "path/to/video.mp4"
   model: "llama3-70b-8192"
   seconds_per_frame: 5
   max_frames: 200
   ```

## Usage

To generate the video summary, run:

```bash
python main.py
```

The script will process the video, generate descriptions for the frames, and use the LLaMA 3 API to produce a summary of the video's content.

## Directory Structure

```
video-content-generator/
├── src/
│   ├── __init__.py
│   ├── process_video.py
│   ├── summarize.py
├── config/
│   ├── config.yaml
├── imagenet-classes.txt
├── README.md
├── requirements.txt
├── main.py
```

- `src/process_video.py`: Contains the function to process the video and extract frames.
- `src/summarize.py`: Contains the function to summarize the frame descriptions using the LLaMA 3 API.
- `config/config.yaml`: Configuration file for API keys and video path.
- `imagenet-classes.txt`: Text file containing ImageNet class labels.
- `main.py`: Main script to run the video summary generation.

## License

This project is licensed under the MIT License.
