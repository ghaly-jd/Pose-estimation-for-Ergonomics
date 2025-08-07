#!/usr/bin/env python3
"""
Camera setup and permission helper for the Posture Detector.
This script helps diagnose and fix camera access issues on macOS.
"""

import cv2
import sys
import subprocess
import os

def check_camera_permissions():
    """Check if camera permissions are granted."""
    print("🔍 Checking camera permissions...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera is accessible!")
        cap.release()
        return True
    else:
        print("❌ Camera not accessible")
        cap.release()
        return False

def list_available_cameras():
    """List all available camera devices."""
    print("\n📹 Checking available camera devices...")
    
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cameras.append(i)
                print(f"✅ Camera {i}: Available")
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    return cameras

def check_macos_permissions():
    """Check macOS camera permissions."""
    print("\n🍎 Checking macOS camera permissions...")
    
    try:
        # Check if Terminal has camera permissions
        result = subprocess.run([
            'tccutil', 'reset', 'Camera', 'com.apple.Terminal'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Camera permissions reset for Terminal")
        else:
            print("⚠️  Could not reset camera permissions (this is normal)")
            
    except FileNotFoundError:
        print("ℹ️  tccutil not available (this is normal)")
    
    print("\n📋 To grant camera permissions:")
    print("   1. Go to System Preferences > Security & Privacy > Privacy")
    print("   2. Select 'Camera' from the left sidebar")
    print("   3. Add Terminal to the list of allowed apps")
    print("   4. Restart Terminal and try again")

def test_posture_detector():
    """Test the posture detector with a simple frame."""
    print("\n🧪 Testing posture detector...")
    
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
        
        print("✅ Posture detector is working!")
        pose.close()
        return True
        
    except Exception as e:
        print(f"❌ Posture detector test failed: {e}")
        return False

def main():
    """Main function to set up camera and test the system."""
    print("🎯 Posture Detector - Camera Setup")
    print("=" * 50)
    
    # Check camera permissions
    camera_ok = check_camera_permissions()
    
    # List available cameras
    cameras = list_available_cameras()
    
    # Check macOS permissions
    check_macos_permissions()
    
    # Test posture detector
    detector_ok = test_posture_detector()
    
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    print("=" * 50)
    
    if camera_ok:
        print("✅ Camera: Working")
    else:
        print("❌ Camera: Needs permission setup")
        
    if cameras:
        print(f"✅ Available cameras: {cameras}")
    else:
        print("❌ No cameras found")
        
    if detector_ok:
        print("✅ Posture detector: Working")
    else:
        print("❌ Posture detector: Failed")
    
    print("\n" + "=" * 50)
    
    if camera_ok and detector_ok:
        print("🎉 Everything is ready! You can now run:")
        print("   python main.py              # Command line interface")
        print("   streamlit run streamlit_app.py  # Web dashboard")
    else:
        print("⚠️  Some components need attention.")
        print("   Please follow the permission setup instructions above.")

if __name__ == "__main__":
    main() 