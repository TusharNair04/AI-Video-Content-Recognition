import yaml
import logging
from src.process_video import process_video
from src.summarize import summarize_video

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configuration
    config = load_config()

    llama_api_key = config['llama_api_key']
    video_path = config['video_path']
    model = config['model']
    seconds_per_frame = config['seconds_per_frame']
    max_frames = config['max_frames']

    # Process video to get frames and descriptions
    base64Frames, frame_descriptions = process_video(video_path, seconds_per_frame, max_frames)

    if base64Frames is None or frame_descriptions is None:
        logging.error("Failed to process video.")
        return

    # Summarize video content
    summary = summarize_video(llama_api_key, frame_descriptions, model)

    print(summary)

if __name__ == "__main__":
    main()
