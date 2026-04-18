import cv2
import numpy as np
import torch
import os
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

def base64_to_frame(base64_string):
    """Convert base64 string to OpenCV frame"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def resize_frame(frame, target_size=(224, 224)):
    """Resize frame to target size"""
    return cv2.resize(frame, target_size)

def normalize_frame_values(frame):
    """Normalize frame pixel values to [0, 1]"""
    return frame.astype(np.float32) / 255.0

def denormalize_frame_values(frame):
    """Denormalize frame pixel values to [0, 255]"""
    return (frame * 255).astype(np.uint8)

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def save_frame(frame, filepath):
    """Save frame to file"""
    create_directory(os.path.dirname(filepath))
    cv2.imwrite(filepath, frame)

def load_frame(filepath):
    """Load frame from file"""
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info

def extract_video_frames(video_path, output_dir, max_frames=None, skip_frames=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    create_directory(output_dir)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return saved_count

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
    return flow

def draw_bounding_box(frame, bbox, label="", confidence=None, color=(0, 255, 0)):
    """Draw bounding box on frame"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    if label or confidence:
        text = label
        if confidence:
            text += f" {confidence:.2f}"
        
        cv2.putText(frame, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def smooth_predictions(predictions, window_size=5):
    """Smooth predictions using moving average"""
    if len(predictions) < window_size:
        return predictions
    
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(predictions), i + window_size // 2 + 1)
        window = predictions[start_idx:end_idx]
        smoothed.append(sum(window) / len(window))
    
    return smoothed

def log_message(message, log_file="system.log"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    print(log_entry)
    
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

def save_json(data, filepath):
    """Save data to JSON file"""
    create_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def get_device():
    """Get the best available device (CUDA/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def validate_video_file(video_path):
    """Validate if video file is readable"""
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Cannot read video frames"
    
    return True, "Valid video file"

def create_video_writer(output_path, fps, width, height, codec='mp4v'):
    """Create video writer object"""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

class PerformanceTimer:
    """Simple performance timer context manager"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.3f} seconds")

import time