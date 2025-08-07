#!/usr/bin/env python3
"""
Test script to verify installation and dependencies for the Posture Detector.
Run this before using the main application.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'cv2',
        'mediapipe',
        'numpy',
        'streamlit',
        'pandas',
        'plotly'
    ]
    
    print("🔍 Testing package imports...")
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_camera():
    """Test if camera is accessible."""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is accessible")
                cap.release()
                return True
            else:
                print("❌ Camera opened but can't read frames")
                cap.release()
                return False
        else:
            print("❌ Camera not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe pose detection."""
    print("\n🤖 Testing MediaPipe pose detection...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize pose detection
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process dummy image
        results = pose.process(dummy_image)
        
        print("✅ MediaPipe pose detection initialized successfully")
        pose.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit availability."""
    print("\n🌐 Testing Streamlit...")
    
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        return True
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Posture Detector - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe Pose", test_mediapipe),
        ("Streamlit", test_streamlit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! Your system is ready to run the Posture Detector.")
        print("\n🚀 You can now run:")
        print("   python main.py              # Command line interface")
        print("   streamlit run streamlit_app.py  # Web dashboard")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Troubleshooting tips:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Check camera permissions")
        print("   3. Ensure Python 3.8+ is installed")
        print("   4. Try restarting your computer")

if __name__ == "__main__":
    main() 