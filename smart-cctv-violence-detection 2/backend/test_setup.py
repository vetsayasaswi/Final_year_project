#!/usr/bin/env python3
"""
Test script to verify the complete Smart CCTV Violence Detection System setup
"""

import sys
import os
import importlib.util
import torch
import cv2
import numpy as np

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    required_modules = [
        'torch', 'torchvision', 'cv2', 'ultralytics', 
        'fastapi', 'uvicorn', 'numpy', 'PIL', 'sklearn'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_project_structure():
    """Test project directory structure"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'preprocessing', 'detection', 'feature_extraction',
        'temporal_model', 'training', 'inference', 
        'alert_system', 'utils', 'models'
    ]
    
    required_files = [
        'app.py', 'config.py', 'requirements.txt',
        'preprocessing/video_to_frames.py',
        'preprocessing/frame_sampling.py',
        'preprocessing/normalization.py',
        'detection/yolo_person_detector.py',
        'feature_extraction/mobilenetv2.py',
        'temporal_model/lstm_model.py',
        'training/train.py',
        'training/evaluate.py',
        'inference/realtime_detection.py',
        'alert_system/alert_service.py',
        'utils/helpers.py'
    ]
    
    missing_items = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Missing directory: {directory}")
            missing_items.append(directory)
        else:
            print(f"✅ Directory: {directory}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            missing_items.append(file_path)
        else:
            print(f"✅ File: {file_path}")
    
    return len(missing_items) == 0

def test_model_components():
    """Test individual model components"""
    print("\nTesting model components...")
    
    try:
        # Test YOLO detector
        from detection.yolo_person_detector import YOLOPersonDetector
        detector = YOLOPersonDetector()
        print("✅ YOLO Person Detector")
        
        # Test MobileNetV2 feature extractor
        from feature_extraction.mobilenetv2 import MobileNetV2FeatureExtractor
        feature_extractor = MobileNetV2FeatureExtractor()
        print("✅ MobileNetV2 Feature Extractor")
        
        # Test LSTM model
        from temporal_model.lstm_model import ViolenceDetectionModel
        model = ViolenceDetectionModel()
        print("✅ Violence Detection Model")
        
        # Test with dummy data
        dummy_input = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Model forward pass: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing components"""
    print("\nTesting preprocessing...")
    
    try:
        from preprocessing.normalization import get_transforms, normalize_frame
        
        # Test transforms
        transform = get_transforms()
        print("✅ Image transforms")
        
        # Test frame normalization
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        normalized = normalize_frame(dummy_frame)
        print(f"✅ Frame normalization: {normalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_alert_system():
    """Test alert system"""
    print("\nTesting alert system...")
    
    try:
        from alert_system.alert_service import AlertService
        
        alert_service = AlertService("test_alerts.json")
        
        # Test alert triggering
        alert = alert_service.trigger_alert(0.85, "test_source")
        if alert:
            print("✅ Alert triggering")
        
        # Test statistics
        stats = alert_service.get_alert_statistics()
        print(f"✅ Alert statistics: {stats}")
        
        # Clean up test file
        if os.path.exists("test_alerts.json"):
            os.remove("test_alerts.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")
    
    try:
        from utils.helpers import (
            frame_to_base64, base64_to_frame, resize_frame,
            get_device, count_parameters
        )
        
        # Test frame conversion
        dummy_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        base64_str = frame_to_base64(dummy_frame)
        recovered_frame = base64_to_frame(base64_str)
        print("✅ Frame base64 conversion")
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Smart CCTV Violence Detection System - Setup Verification")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Components", test_model_components),
        ("Preprocessing", test_preprocessing),
        ("Alert System", test_alert_system),
        ("Utilities", test_utilities)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place RWF-2000 dataset in dataset/RWF-2000/")
        print("2. Run preprocessing scripts")
        print("3. Train the model")
        print("4. Start the backend server")
        print("5. Launch the frontend application")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        print("Refer to README.md for detailed setup instructions.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)