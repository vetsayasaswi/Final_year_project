#!/usr/bin/env python3
"""
Quick start script for Smart CCTV Violence Detection System
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process"""
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    # Check Python packages
    required_packages = ['torch', 'fastapi', 'opencv-python', 'ultralytics']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing Python packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Node.js not found. Please install Node.js 16+")
            return False
    except FileNotFoundError:
        print("Node.js not found. Please install Node.js 16+")
        return False
    
    print("✅ All dependencies found")
    return True

def setup_frontend():
    """Setup frontend if not already done"""
    frontend_dir = Path("../frontend")
    node_modules = frontend_dir / "node_modules"
    
    if not node_modules.exists():
        print("Installing frontend dependencies...")
        process = run_command("npm install", cwd=frontend_dir)
        if process:
            process.wait()
            if process.returncode == 0:
                print("✅ Frontend dependencies installed")
            else:
                print("❌ Failed to install frontend dependencies")
                return False
    
    return True

def start_backend():
    """Start the backend server"""
    print("Starting backend server...")
    
    # Check if model exists
    models_dir = Path("models")
    model_files = list(models_dir.glob("best_model_*.pth")) if models_dir.exists() else []
    
    if not model_files:
        print("⚠️  No trained model found. Some features will be limited.")
        print("To train a model:")
        print("1. Place RWF-2000 dataset in dataset/RWF-2000/")
        print("2. Run: python preprocessing/video_to_frames.py")
        print("3. Run: python preprocessing/frame_sampling.py")
        print("4. Run: python preprocessing/normalization.py")
        print("5. Run: python training/train.py")
    
    # Start FastAPI server
    backend_process = run_command("python app.py")
    
    if backend_process:
        print("✅ Backend server starting on http://localhost:8000")
        return backend_process
    else:
        print("❌ Failed to start backend server")
        return None

def start_frontend():
    """Start the frontend application"""
    print("Starting frontend application...")
    
    frontend_dir = Path("../frontend")
    frontend_process = run_command("npm start", cwd=frontend_dir)
    
    if frontend_process:
        print("✅ Frontend application starting on http://localhost:3000")
        return frontend_process
    else:
        print("❌ Failed to start frontend application")
        return None

def monitor_processes(processes):
    """Monitor running processes"""
    try:
        while True:
            time.sleep(1)
            for name, process in processes.items():
                if process and process.poll() is not None:
                    print(f"⚠️  {name} process stopped")
                    return
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        for name, process in processes.items():
            if process:
                process.terminate()
                print(f"✅ {name} stopped")

def main():
    """Main function to start the system"""
    print("🚀 Smart CCTV Violence Detection System - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        return False
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return False
    
    # Wait a moment for backend to start
    print("Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        if backend_process:
            backend_process.terminate()
        return False
    
    print("\n" + "="*60)
    print("🎉 System started successfully!")
    print("📊 Backend API: http://localhost:8000")
    print("🌐 Frontend App: http://localhost:3000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    print("="*60)
    
    # Monitor processes
    processes = {
        "Backend": backend_process,
        "Frontend": frontend_process
    }
    
    monitor_processes(processes)
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)