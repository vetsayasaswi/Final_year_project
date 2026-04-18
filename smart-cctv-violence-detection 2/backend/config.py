import os
import torch
# Dataset paths
DATASET_ROOT = "dataset/RWF-2000"
TRAIN_VIOLENCE_PATH = os.path.join(DATASET_ROOT, "train", "Fight")
TRAIN_NONVIOLENCE_PATH = os.path.join(DATASET_ROOT, "train", "NonFight")

TEST_VIOLENCE_PATH = os.path.join(DATASET_ROOT, "test", "Fight")
TEST_NONVIOLENCE_PATH = os.path.join(DATASET_ROOT, "test", "NonFight")

# Preprocessing parameters
FRAME_SIZE = (224, 224)
SEQUENCE_LENGTH = 16
FPS = 10
BATCH_SIZE = 8

# Model parameters
MOBILENET_FEATURES = 1280
LSTM_HIDDEN_SIZE = 256
NUM_CLASSES = 2
DROPOUT_RATE = 0.5

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 15
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Alert system
VIOLENCE_THRESHOLD = 0.8
ALERT_COOLDOWN = 5  # seconds

# Paths
MODEL_SAVE_PATH = "models/violence_detection_model.pth"
YOLO_MODEL_PATH = "yolov8n.pt"

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"