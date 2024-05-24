import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import base64
import logging

def process_video(video_path, seconds_per_frame=5, max_frames=500):
    logging.info(f"Processing video: {video_path}")
    base64Frames = []
    frame_descriptions = []
    base_video_path, _ = os.path.splitext(video_path)

    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame = 0

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        labels = open("imagenet-classes.txt").read().splitlines()

        while curr_frame < total_frames - 1 and len(base64Frames) < max_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

            description = ', '.join([labels[catid] for catid in top5_catid])
            frame_descriptions.append(description)

            curr_frame += frames_to_skip
        video.release()

        logging.info(f"Extracted {len(base64Frames)} frames")
        return base64Frames, frame_descriptions

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return None, None
