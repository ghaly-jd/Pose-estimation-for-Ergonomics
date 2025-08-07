#!/usr/bin/env python3
"""
Demo recorder for the Posture Detector.
Records posture detection sessions for demonstration purposes.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os
from typing import Tuple

class PostureDetectorRecorder:
    def __init__(self, output_path: str = "demo_video.mp4"):
        """Initialize the posture detector with recording capabilities."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Posture thresholds
        self.NECK_ANGLE_THRESHOLD = 45
        self.SHOULDER_ANGLE_THRESHOLD = 15
        
        # Recording setup
        self.output_path = output_path
        self.video_writer = None
        self.frame_count = 0
        
        # Colors for feedback
        self.GOOD_COLOR = (0, 255, 0)  # Green
        self.BAD_COLOR = (0, 0, 255)   # Red
        self.WARNING_COLOR = (0, 165, 255)  # Orange
        
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate the angle between three points."""
        if not all(point1) or not all(point2) or not all(point3):
            return 0
            
        # Convert to numpy arrays
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def analyze_posture(self, landmarks) -> Tuple[str, float, float]:
        """Analyze posture based on key landmarks."""
        if not landmarks:
            return "Unknown", 0, 0
            
        # Get key landmarks
        left_shoulder = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
        right_shoulder = (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
        left_ear = (landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y)
        left_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y)
        
        # Calculate angles
        neck_angle = self.calculate_angle(left_ear, left_shoulder, left_hip)
        shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, 
                                           (right_shoulder[0] + 1, right_shoulder[1]))
        
        # Determine posture state
        if neck_angle > self.NECK_ANGLE_THRESHOLD or shoulder_angle > self.SHOULDER_ANGLE_THRESHOLD:
            posture_state = "Slouching"
        else:
            posture_state = "Good Posture"
            
        return posture_state, neck_angle, shoulder_angle
    
    def draw_posture_feedback(self, image, posture_state: str, 
                            neck_angle: float, shoulder_angle: float):
        """Draw posture feedback on the image."""
        h, w, _ = image.shape
        
        # Choose color based on posture state
        if posture_state == "Good Posture":
            color = self.GOOD_COLOR
        elif posture_state == "Slouching":
            color = self.BAD_COLOR
        else:
            color = self.WARNING_COLOR
        
        # Draw posture state text
        cv2.putText(image, f"Posture: {posture_state}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw angle information
        cv2.putText(image, f"Neck Angle: {neck_angle:.1f}°", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.1f}°", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw recording indicator
        cv2.putText(image, "REC", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(image, "Press 'q' to quit, 'r' to start/stop recording", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_posture_lines(self, image, landmarks):
        """Draw posture analysis lines on the image."""
        if not landmarks:
            return
            
        h, w, _ = image.shape
        
        # Get key points
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        left_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * w),
                   int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
        left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                   int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        
        # Draw neck line (ear to shoulder)
        cv2.line(image, left_ear, left_shoulder, (255, 255, 0), 3)
        
        # Draw spine line (shoulder to hip)
        cv2.line(image, left_shoulder, left_hip, (255, 255, 0), 3)
        
        # Draw shoulder line
        cv2.line(image, left_shoulder, right_shoulder, (0, 255, 255), 3)
    
    def start_recording(self, frame_width: int, frame_height: int):
        """Start recording video."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 30.0, (frame_width, frame_height))
        print(f"🎬 Started recording to {self.output_path}")
    
    def stop_recording(self):
        """Stop recording video."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print(f"✅ Recording saved to {self.output_path}")
    
    def record_frame(self, frame):
        """Record a frame to video."""
        if self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1
    
    def run_demo_recorder(self):
        """Run the demo recorder."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("🎯 Posture Detector Demo Recorder Started!")
        print("📋 Instructions:")
        print("   - Sit at your desk")
        print("   - Press 'r' to start/stop recording")
        print("   - Press 'q' to quit")
        print("   - Green = Good Posture")
        print("   - Red = Slouching")
        
        is_recording = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Analyze posture
                posture_state, neck_angle, shoulder_angle = self.analyze_posture(results.pose_landmarks.landmark)
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Draw posture feedback
                self.draw_posture_feedback(frame, posture_state, neck_angle, shoulder_angle)
                
                # Draw posture lines
                self.draw_posture_lines(frame, results.pose_landmarks.landmark)
            
            # Record frame if recording
            if is_recording:
                self.record_frame(frame)
            
            # Display the frame
            cv2.imshow('Posture Detector Demo Recorder', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not is_recording:
                    # Start recording
                    self.start_recording(frame_width, frame_height)
                    is_recording = True
                else:
                    # Stop recording
                    self.stop_recording()
                    is_recording = False
        
        # Clean up
        if is_recording:
            self.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

def main():
    """Main function to run the demo recorder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Record posture detection demo")
    parser.add_argument("--output", "-o", default="demo_video.mp4", 
                       help="Output video file path")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    detector = PostureDetectorRecorder(args.output)
    detector.run_demo_recorder()

if __name__ == "__main__":
    main() 