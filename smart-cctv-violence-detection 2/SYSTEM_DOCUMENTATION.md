# Smart CCTV Violence Detection System - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Architecture](#project-architecture)
3. [Dataset Explanation](#dataset-explanation)
4. [Core Concepts](#core-concepts)
5. [Implementation Details](#implementation-details)
6. [Implemented Features](#implemented-features)
7. [System Workflow](#system-workflow)

---

## System Overview

### What This System Does
This is an AI-powered surveillance system that automatically detects violent behavior in video streams. It analyzes CCTV footage in real-time and triggers alerts when violence is detected.

### Technology Stack
- **Backend**: Python, PyTorch, FastAPI
- **Frontend**: React.js
- **Deep Learning**: MobileNetV2 (CNN) + LSTM (RNN)
- **Object Detection**: YOLOv8
- **Computer Vision**: OpenCV

### System Purpose
Designed for automated monitoring of surveillance footage to detect fights, physical altercations, and violent behavior without requiring constant human supervision.

---

## Project Architecture

### Directory Structure

```
smart-cctv-violence-detection/
│
├── backend/
│   ├── dataset/RWF-2000/              # Raw video files
│   │   ├── train/Fight/               # Training violence videos
│   │   ├── train/NonFight/            # Training non-violence videos
│   │   ├── test/Fight/                # Test violence videos
│   │   └── test/NonFight/             # Test non-violence videos
│   │
│   ├── frames/                        # Extracted frames (generated)
│   │   ├── train/Violence/
│   │   ├── train/NonViolence/
│   │   ├── test/Violence/
│   │   └── test/NonViolence/
│   │
│   ├── sequences/                     # Frame sequence metadata (generated)
│   │   ├── train_sequences.json
│   │   └── test_sequences.json
│   │
│   ├── models/                        # Trained models (generated)
│   │   └── best_model.pth
│   │
│   ├── preprocessing/                 # Data preparation scripts
│   │   ├── video_to_frames.py        # Extract frames from videos
│   │   ├── frame_sampling.py         # Create frame sequences
│   │   └── normalization.py          # Data normalization & loading
│   │
│   ├── detection/
│   │   └── yolo_person_detector.py   # Person detection using YOLO
│   │
│   ├── feature_extraction/
│   │   └── mobilenetv2.py            # CNN feature extractor
│   │
│   ├── temporal_model/
│   │   └── lstm_model.py             # LSTM temporal model
│   │
│   ├── training/
│   │   ├── train.py                  # Model training script
│   │   └── evaluate.py               # Model evaluation
│   │
│   ├── inference/
│   │   ├── realtime_detection.py     # Real-time detection
│   │   └── live_video_demo.py        # Demo script
│   │
│   ├── alert_system/
│   │   └── alert_service.py          # Alert management
│   │
│   ├── utils/
│   │   └── helpers.py                # Helper functions
│   │
│   ├── config.py                     # Configuration parameters
│   ├── app.py                        # FastAPI backend server
│   └── requirements.txt              # Python dependencies
│
└── frontend/                          # React web application
    ├── src/
    │   ├── App.js                    # Main React component
    │   ├── App.css                   # Styling
    │   └── index.js                  # Entry point
    ├── public/
    └── package.json                  # Node dependencies
```

### Three-Stage Pipeline Architecture

```
Stage 1: Person Detection (Optional)
    ↓
Stage 2: Spatial Feature Extraction (MobileNetV2)
    ↓
Stage 3: Temporal Pattern Analysis (LSTM)
    ↓
Violence Probability Output
```

---

## Dataset Explanation

### RWF-2000 Dataset

**Full Name**: Real-World Fight Dataset 2000

**Purpose**: A benchmark dataset specifically created for violence detection in surveillance videos.

**Dataset Characteristics**:
- **Total Videos**: ~2000 video clips
- **Source**: Real-world CCTV footage
- **Categories**: Fight (Violence) and NonFight (Non-Violence)
- **Split**: Training set and Test set
- **Environment**: Various indoor and outdoor locations
- **Quality**: Variable (simulating real surveillance conditions)
- **Duration**: Videos of varying lengths (typically 2-10 seconds)

**Dataset Structure**:
```
RWF-2000/
├── train/
│   ├── Fight/          # Videos containing violent behavior
│   │   ├── video_001.avi
│   │   ├── video_002.avi
│   │   └── ...
│   └── NonFight/       # Videos with normal behavior
│       ├── video_001.avi
│       ├── video_002.avi
│       └── ...
└── test/
    ├── Fight/
    └── NonFight/
```

**What Constitutes "Fight" (Violence)**:
- Physical altercations between people
- Punching, kicking, pushing
- Aggressive physical contact
- Wrestling or grappling

**What Constitutes "NonFight" (Non-Violence)**:
- Normal walking and standing
- Conversations
- Handshakes or friendly gestures
- Sports activities (non-contact)
- Dancing or exercising

**Dataset Challenges**:
- Variable camera angles
- Different lighting conditions
- Occlusions (people blocking each other)
- Crowded scenes
- Low resolution (typical of CCTV)

**Why This Dataset**:
- Realistic surveillance scenarios
- Balanced classes (roughly equal Fight/NonFight samples)
- Diverse environments
- Standard benchmark for violence detection research

---

## Core Concepts

### 1. Temporal Modeling

**Concept**: Violence is not a single frame event - it's a sequence of actions over time.

**Why It Matters**:
- A raised fist in one frame could be waving or punching
- Context from previous and subsequent frames is essential
- Motion patterns distinguish violence from normal activity

**Implementation**: Using 16 consecutive frames (1.6 seconds at 10 FPS) to capture the temporal evolution of actions.

### 2. Transfer Learning

**Concept**: Using knowledge learned from one task to improve performance on another task.

**In This System**:
- MobileNetV2 is pretrained on ImageNet (1.4M images, 1000 categories)
- It already knows how to detect edges, textures, objects, and poses
- We fine-tune it for violence detection instead of training from scratch

**Benefits**:
- Faster training (hours instead of days)
- Better accuracy with limited data
- Requires less computational resources

### 3. Feature Extraction

**Concept**: Converting raw pixel data into meaningful numerical representations.

**Process**:
- Input: 224×224×3 image (150,528 numbers)
- Output: 1280-dimensional feature vector (1,280 numbers)
- These features encode high-level information (poses, objects, scene context)

**Why Compress**:
- Removes redundant information
- Focuses on relevant patterns
- Makes temporal modeling computationally feasible

### 4. Recurrent Neural Networks (LSTM)

**Concept**: Neural networks designed to process sequences by maintaining memory of previous inputs.

**LSTM Specifically**:
- **Long Short-Term Memory**: Can remember information across many timesteps
- **Cell State**: Acts as a "memory highway" carrying information forward
- **Gates**: Control what information to remember, forget, or output

**Why LSTM for Violence**:
- Understands temporal dependencies (what happened before affects what happens now)
- Captures motion dynamics (acceleration, deceleration)
- Learns action sequences (approach → confrontation → strike)

### 5. Bidirectional Processing

**Concept**: Processing sequences in both forward and backward directions.

**Forward Pass**: Frames 0→1→2→...→15 (what leads to this moment)
**Backward Pass**: Frames 15→14→13→...→0 (what follows this moment)

**Combined Understanding**: The model sees both past and future context, improving accuracy.

### 6. Binary Classification

**Concept**: Categorizing inputs into one of two classes.

**In This System**:
- Class 0: Non-Violence
- Class 1: Violence

**Output**: A probability between 0.0 and 1.0
- 0.0 = Definitely non-violent
- 0.5 = Uncertain
- 1.0 = Definitely violent

### 7. Sliding Window

**Concept**: Continuously updating a fixed-size buffer of recent frames.

**How It Works**:
```
Frame 1 arrives  → Buffer: [1]
Frame 2 arrives  → Buffer: [1, 2]
...
Frame 16 arrives → Buffer: [1, 2, ..., 16] → Run prediction
Frame 17 arrives → Buffer: [2, 3, ..., 17] → Run prediction
```

**Benefit**: Enables continuous real-time monitoring without reprocessing entire videos.

### 8. Threshold-Based Alerting

**Concept**: Triggering actions only when confidence exceeds a predefined level.

**In This System**:
- Threshold: 0.7 (70% confidence)
- Below 0.7: Monitor but don't alert
- Above 0.7: Trigger alert

**Why 0.7**:
- Balances sensitivity (catching real violence) vs. specificity (avoiding false alarms)
- Configurable based on deployment requirements

### 9. Alert Cooldown

**Concept**: Preventing repeated alerts for the same incident.

**Implementation**: 5-second minimum gap between alerts

**Why Needed**:
- A single fight might trigger high probabilities for 20+ consecutive frames
- Without cooldown: 20 alerts for one incident
- With cooldown: 1 alert per incident

---

## Implementation Details

### Configuration Parameters (config.py)

**Dataset Paths**:
- `DATASET_ROOT`: Location of RWF-2000 videos
- Separate paths for train/test and Fight/NonFight

**Preprocessing Parameters**:
- `FRAME_SIZE`: (224, 224) - Input dimensions for MobileNetV2
- `SEQUENCE_LENGTH`: 16 - Number of frames per sequence
- `FPS`: 10 - Frame extraction rate
- `BATCH_SIZE`: 8 - Sequences processed simultaneously

**Model Parameters**:
- `MOBILENET_FEATURES`: 1280 - Feature vector dimension
- `LSTM_HIDDEN_SIZE`: 256 - LSTM hidden units
- `NUM_CLASSES`: 2 - Binary classification
- `DROPOUT_RATE`: 0.5 - Regularization strength

**Training Parameters**:
- `LEARNING_RATE`: 0.001 - Initial learning rate
- `EPOCHS`: 15 - Training iterations
- `DEVICE`: "cuda" or "cpu" - Automatic GPU detection

**Alert System**:
- `VIOLENCE_THRESHOLD`: 0.7 - Alert trigger level
- `ALERT_COOLDOWN`: 5 seconds - Minimum gap between alerts

### Data Preprocessing Pipeline

**Step 1: Video to Frames (video_to_frames.py)**

**Input**: Video files (.avi, .mp4, .mov)
**Output**: Individual frame images (.jpg)

**Process**:
1. Open video file
2. Read original FPS (e.g., 30 fps)
3. Calculate frame interval: original_fps / target_fps (e.g., 30/10 = 3)
4. Extract every Nth frame (e.g., every 3rd frame)
5. Save as JPEG images

**Why Fixed FPS**:
- Videos have different frame rates (24, 30, 60 fps)
- Model needs consistent temporal sampling
- 10 FPS captures sufficient motion without redundancy

**Output Structure**:
```
frames/train/Violence/video_001/
    ├── frame_000000.jpg
    ├── frame_000001.jpg
    ├── frame_000002.jpg
    └── ...
```

**Step 2: Frame Sequence Creation (frame_sampling.py)**

**Input**: Extracted frame images
**Output**: JSON files with sequence metadata

**Process**:
1. List all frames for each video
2. Group into sequences of 16 consecutive frames
3. Use stride of 8 (50% overlap)
4. Handle short videos by repeating frames
5. Assign labels (1 for Violence, 0 for NonViolence)
6. Save sequence information as JSON

**Sequence Format**:
```json
{
  "frames": [
    "frames/train/Violence/video_001/frame_000000.jpg",
    "frames/train/Violence/video_001/frame_000001.jpg",
    ...
    "frames/train/Violence/video_001/frame_000015.jpg"
  ],
  "label": 1
}
```

**Why Overlapping Sequences**:
- Increases training data diversity
- Ensures violent events at boundaries aren't missed
- Provides multiple perspectives of the same action

**Step 3: Normalization (normalization.py)**

**Input**: Sequence metadata JSON
**Output**: PyTorch DataLoader

**Transformations Applied**:
1. Load image from disk
2. Convert BGR to RGB (OpenCV uses BGR)
3. Resize to 224×224
4. Convert to tensor (0-1 range)
5. Normalize with ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

**Why ImageNet Normalization**:
- MobileNetV2 was trained with these statistics
- Ensures pretrained features work correctly
- Standard practice for transfer learning

**DataLoader Configuration**:
- Batch size: 8 sequences
- Shuffle: True (training), False (testing)
- Parallel workers: 2
- Pin memory: True (faster GPU transfer)

**Output Tensor Shape**: [8, 16, 3, 224, 224]
- 8 = batch size
- 16 = sequence length
- 3 = RGB channels
- 224×224 = frame dimensions

### Model Architecture

**Component 1: MobileNetV2 Feature Extractor (mobilenetv2.py)**

**Purpose**: Extract spatial features from individual frames

**Architecture**:
- Pretrained on ImageNet
- Remove classification head
- Keep convolutional backbone
- Add global average pooling

**Layer Freezing**:
- First 10 layers: Frozen (not trainable)
- Remaining layers: Trainable

**Why Freeze Early Layers**:
- Early layers detect universal features (edges, textures)
- Prevents overfitting on small dataset
- Speeds up training

**Input**: [Batch×Time, 3, 224, 224]
**Output**: [Batch×Time, 1280]

**Processing Strategy**:
- Merge batch and time dimensions
- Process all frames in parallel
- Restore time dimension after extraction

**Component 2: LSTM Temporal Model (lstm_model.py)**

**Purpose**: Analyze temporal patterns across frame sequence

**Architecture**:
- 2-layer Bidirectional LSTM
- 256 hidden units per direction
- Dropout: 0.5
- Fully connected classifier

**LSTM Configuration**:
- Input size: 1280 (MobileNetV2 features)
- Hidden size: 256
- Bidirectional: Yes (512 total hidden units)
- Batch first: Yes

**Classifier**:
- Dropout (0.5)
- Linear (512 → 256)
- ReLU activation
- Dropout (0.5)
- Linear (256 → 1)

**Input**: [Batch, 16, 1280]
**Output**: [Batch, 1] (single logit)

**Why Single Output**:
- Binary classification needs one value
- Positive logit = Violence
- Negative logit = Non-violence

**Component 3: Complete Model (ViolenceDetectionModel)**

**Combined Pipeline**:
1. Input video sequence
2. MobileNetV2 extracts features
3. LSTM analyzes temporal patterns
4. Classifier outputs violence logit
5. Sigmoid converts to probability

**Forward Pass**:
```
[B, 16, 3, 224, 224] 
    → MobileNetV2 → 
[B, 16, 1280] 
    → LSTM → 
[B, 1] logit 
    → Sigmoid → 
[B, 1] probability
```

### Training Process (train.py)

**Loss Function**: BCEWithLogitsLoss
- Binary Cross-Entropy with Logits
- Combines sigmoid and BCE for numerical stability
- Standard for binary classification

**Optimizer**: Adam
- Adaptive learning rate per parameter
- Learning rate: 0.001
- Momentum and RMSprop combined

**Learning Rate Schedule**:
- Every 10 epochs: multiply LR by 0.5
- Epoch 1-10: LR = 0.001
- Epoch 11-20: LR = 0.0005
- Epoch 21-30: LR = 0.00025

**Training Loop**:
1. Load batch of sequences
2. Forward pass through model
3. Calculate loss
4. Backward pass (compute gradients)
5. Update weights
6. Repeat for all batches

**Evaluation Loop**:
1. Set model to eval mode
2. Disable gradient computation
3. Process test batches
4. Calculate accuracy
5. No weight updates

**Checkpoint Saving**:
- Save model when test accuracy improves
- Stores: model weights, optimizer state, training history
- File: models/best_model.pth

**Expected Performance**:
- Training accuracy: 80-85%
- Test accuracy: 75-82%
- Training time: 2-4 hours (GPU)

### Inference System (realtime_detection.py)

**Model Loading**:
1. Initialize ViolenceDetectionModel
2. Load saved checkpoint
3. Set to evaluation mode
4. Move to GPU/CPU

**Frame Buffer**:
- Circular buffer holding last 16 frames
- Automatically removes oldest when adding new
- Maintains sliding window

**Processing New Frame**:
1. Optional: YOLO person detection
2. Normalize frame (same as training)
3. Add to buffer
4. If buffer has 16 frames: run prediction
5. Return violence probability

**Prediction Steps**:
1. Stack 16 frames into tensor
2. Add batch dimension
3. Forward pass through model
4. Apply sigmoid to logit
5. Extract probability value

**Alert Logic**:
1. Check if probability > 0.7
2. Check if 5 seconds since last alert
3. If both true: trigger alert
4. Save alert to database
5. Send notifications

### Alert System (alert_service.py)

**Alert Data Structure**:
- ID: Unique identifier
- Timestamp: When alert occurred
- Confidence: Violence probability
- Source: Camera/video identifier
- Frame data: Optional image

**Alert Storage**:
- JSON file: alerts.json
- Log file: violence_alerts.log
- In-memory list for quick access

**Alert Methods**:
- `trigger_alert()`: Create new alert
- `get_recent_alerts()`: Fetch last N hours
- `get_alert_statistics()`: Summary stats
- `clear_old_alerts()`: Remove old entries

**Notification Placeholders**:
- Email notification (not implemented)
- SMS notification (not implemented)
- Push notification (not implemented)
- Console logging (implemented)

### Backend API (app.py)

**Framework**: FastAPI

**REST Endpoints**:
- `GET /health`: System health check
- `POST /upload-video`: Upload video for processing
- `POST /analyze-frame`: Analyze single frame
- `GET /alerts`: Get recent alerts
- `GET /alerts/statistics`: Get alert stats
- `DELETE /alerts`: Clear old alerts
- `GET /model/info`: Model information
- `GET /system/status`: System status

**WebSocket Endpoint**:
- `/ws`: Real-time bidirectional communication
- Receives frames from frontend
- Sends analysis results back
- Broadcasts alerts to all clients

**CORS Configuration**:
- Allows all origins (development mode)
- Should be restricted in production

**Startup Process**:
1. Find trained model file
2. Initialize RealTimeViolenceDetector
3. Load model weights
4. Ready to accept requests

### Frontend Application (App.js)

**Technology**: React 18.2.0

**Key Libraries**:
- axios: HTTP requests
- socket.io-client: WebSocket communication
- react-webcam: Camera access

**Main Components**:
1. Webcam monitoring section
2. Video upload section
3. Statistics dashboard
4. Alert history panel

**Webcam Processing**:
1. User clicks "Start Webcam"
2. Request camera permission
3. Capture frame every 1 second
4. Convert to base64
5. Send via WebSocket
6. Receive violence probability
7. Update UI overlay

**Video Upload**:
1. User selects video file
2. Upload via HTTP POST
3. Backend processes video
4. Return analysis results
5. Display summary

**Real-Time Updates**:
- WebSocket connection maintained
- Listen for alert events
- Update UI immediately
- Show browser notifications

**UI Display**:
- Live webcam feed with overlay
- Violence probability percentage
- Risk level indicator (color-coded)
- Person count
- Alert banner
- Statistics cards
- Alert history list

---

## Implemented Features

### ✅ Core Features (Fully Implemented)

**1. Data Preprocessing**
- Video to frame extraction at fixed FPS
- Frame sequence creation with overlap
- Image normalization and augmentation
- PyTorch DataLoader integration

**2. Deep Learning Model**
- MobileNetV2 feature extraction
- Bidirectional LSTM temporal modeling
- Binary classification
- Transfer learning from ImageNet

**3. Model Training**
- Batch processing
- GPU acceleration
- Learning rate scheduling
- Model checkpointing
- Training history tracking

**4. Model Evaluation**
- Accuracy calculation
- Confusion matrix
- Classification report
- ROC curve analysis

**5. Real-Time Inference**
- Sliding window processing
- Frame buffer management
- Violence probability calculation
- Continuous monitoring

**6. Person Detection**
- YOLOv8 integration
- Bounding box visualization
- Person-focused analysis

**7. Alert System**
- Threshold-based triggering
- Cooldown mechanism
- Alert storage (JSON)
- Alert history
- Statistics calculation

**8. Backend API**
- FastAPI server
- REST endpoints
- WebSocket support
- Video upload handling
- Frame analysis

**9. Frontend Dashboard**
- React web application
- Live webcam monitoring
- Video upload interface
- Real-time updates
- Alert notifications
- Statistics display

**10. Live Demo**
- Video file processing
- Real-time visualization
- Bounding box overlay
- Violence status display

### ❌ Not Implemented

**1. Notification Services**
- Email notifications (placeholder only)
- SMS notifications (placeholder only)
- Push notifications (placeholder only)

**2. User Authentication**
- No login system
- No user management
- No access control

**3. Database Integration**
- No SQL/NoSQL database
- Only JSON file storage

**4. Multi-Camera Support**
- Single camera/video at a time
- No camera management system

**5. Cloud Deployment**
- No cloud infrastructure
- Local deployment only

**6. Advanced Analytics**
- No historical trend analysis
- No predictive analytics
- No heat maps

**7. Video Recording**
- No automatic recording on alert
- No video archival system

---

## System Workflow

### Training Workflow

```
1. Prepare Dataset
   ├── Download RWF-2000 videos
   ├── Organize into Fight/NonFight folders
   └── Place in dataset/RWF-2000/

2. Run Preprocessing
   ├── python preprocessing/video_to_frames.py
   │   └── Extracts frames at 10 FPS
   ├── python preprocessing/frame_sampling.py
   │   └── Creates 16-frame sequences
   └── python preprocessing/normalization.py
       └── Tests DataLoader

3. Train Model
   ├── python training/train.py
   │   ├── Loads sequences
   │   ├── Trains for 15 epochs
   │   ├── Saves best model
   │   └── Generates training_history.png
   └── Output: models/best_model.pth

4. Evaluate Model
   └── python training/evaluate.py
       ├── Loads best model
       ├── Tests on test set
       ├── Generates confusion matrix
       └── Generates ROC curve
```

### Inference Workflow

```
1. Load Trained Model
   ├── Initialize ViolenceDetectionModel
   ├── Load models/best_model.pth
   └── Set to evaluation mode

2. Process Video/Webcam
   ├── Capture frame
   ├── Optional: YOLO person detection
   ├── Normalize frame
   ├── Add to 16-frame buffer
   └── When buffer full: run prediction

3. Generate Prediction
   ├── MobileNetV2 feature extraction
   ├── LSTM temporal analysis
   ├── Sigmoid activation
   └── Output: violence probability (0.0-1.0)

4. Alert Decision
   ├── Check if probability > 0.7
   ├── Check if 5 seconds since last alert
   └── If both true: trigger alert

5. Alert Actions
   ├── Save to alerts.json
   ├── Log to violence_alerts.log
   ├── Send to frontend via WebSocket
   └── Display notification
```

### Production Deployment Workflow

```
1. Start Backend Server
   └── python app.py
       ├── Loads trained model
       ├── Starts FastAPI server
       └── Listens on http://localhost:8000

2. Start Frontend Application
   └── npm start (in frontend/)
       ├── Starts React dev server
       └── Opens http://localhost:3000

3. User Interaction
   ├── Option A: Live Webcam
   │   ├── Click "Start Webcam"
   │   ├── Grant camera permission
   │   ├── Frames sent every 1 second
   │   └── Real-time violence detection
   │
   └── Option B: Video Upload
       ├── Select video file
       ├── Upload to backend
       ├── Backend processes video
       └── Display results

4. Monitoring
   ├── View live violence probability
   ├── See alert history
   ├── Check statistics
   └── Receive browser notifications
```

---

## Key Technical Decisions

### Why 16 Frames?
- Represents 1.6 seconds at 10 FPS
- Sufficient to capture violent actions
- Computationally manageable
- Standard in action recognition research

### Why 10 FPS?
- Balances temporal resolution and computation
- Eliminates redundant consecutive frames
- Consistent across different video sources
- Real-time processing capable

### Why MobileNetV2?
- Lightweight (3.5M parameters)
- Fast inference (suitable for real-time)
- Pretrained on ImageNet
- Good accuracy/speed tradeoff
- Designed for deployment

### Why Bidirectional LSTM?
- Sees both past and future context
- Improves accuracy by 5-10%
- Better understanding of action sequences
- Captures temporal dependencies

### Why Freeze Early CNN Layers?
- Prevents overfitting on small dataset
- Low-level features are universal
- Speeds up training
- Focuses learning on task-specific patterns

### Why 0.7 Threshold?
- Balances sensitivity and specificity
- Reduces false positives
- Configurable based on requirements
- Empirically determined

### Why 5-Second Cooldown?
- Prevents alert spam
- One alert per incident
- Gives time for response
- Reduces notification fatigue

---

## Performance Characteristics

### Model Metrics
- **Parameters**: ~5.5M total
- **Model Size**: ~22 MB
- **Training Accuracy**: 80-85%
- **Test Accuracy**: 75-82%
- **Inference Speed**: 50-100ms per sequence (GPU)
- **Real-time Capable**: Yes (>10 FPS)

### Resource Requirements
- **Training**: 4-6 GB GPU memory
- **Inference**: 1-2 GB GPU memory
- **CPU Mode**: 2-4 GB RAM
- **Storage**: ~10 GB (dataset + frames + models)

### Expected Results
- **Precision**: 75-80%
- **Recall**: 80-85%
- **F1-Score**: 77-82%
- **AUC**: 0.85-0.90

---

## Troubleshooting Guide

### Common Issues

**1. Missing Dataset Directories**
- Error: "Missing directory: dataset/RWF-2000\train/Fight"
- Solution: Create directories or download RWF-2000 dataset

**2. CUDA Out of Memory**
- Error: "RuntimeError: CUDA out of memory"
- Solution: Reduce BATCH_SIZE in config.py

**3. Model Not Found**
- Error: "No trained model found"
- Solution: Run training first (python training/train.py)

**4. Webcam Not Working**
- Error: Camera permission denied
- Solution: Check browser permissions, camera drivers

**5. Import Errors**
- Error: "ModuleNotFoundError"
- Solution: Install requirements (pip install -r requirements.txt)

---

## Future Enhancement Possibilities

### Potential Improvements (Not Currently Implemented)
- Multi-camera support
- Cloud deployment
- Database integration
- User authentication
- Email/SMS notifications
- Video recording on alert
- Advanced analytics dashboard
- Model retraining interface
- Configuration UI
- Mobile application

---

## Conclusion

This system provides a complete end-to-end solution for automated violence detection in surveillance footage. It combines modern deep learning techniques (CNN + LSTM) with practical deployment considerations (real-time processing, alert management, web interface).

The implementation is research-oriented but production-ready for small-scale deployments. It demonstrates the feasibility of AI-powered surveillance systems while maintaining reasonable computational requirements.

**Key Strengths**:
- Real-time processing capability
- Transfer learning for efficiency
- Temporal modeling for accuracy
- Complete web-based interface
- Configurable alert system

**Current Limitations**:
- Single camera/video at a time
- No persistent database
- Limited notification options
- No user management
- Local deployment only

This documentation provides a complete technical understanding of the system architecture, implementation details, and operational characteristics.
