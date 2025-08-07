#!/usr/bin/env python3
"""
Debug version of posture detector that shows detailed angle information.
This will help us calibrate the thresholds properly.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple

class PostureDetectorDebug:
    def __init__(self):
        """Initialize the posture detector with debug information."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Posture thresholds (very lenient for testing)
        self.NECK_ANGLE_THRESHOLD = 80  # degrees (very high)
        self.SHOULDER_ANGLE_THRESHOLD = 40  # degrees (very high)
        
        # Colors for feedback
        self.GOOD_COLOR = (0, 255, 0)  # Green
        self.BAD_COLOR = (0, 0, 255)   # Red
        self.WARNING_COLOR = (0, 165, 255)  # Orange
        self.DEBUG_COLOR = (255, 255, 0)  # Yellow
        
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
        
        # Determine posture state with very lenient thresholds
        if neck_angle > self.NECK_ANGLE_THRESHOLD or shoulder_angle > self.SHOULDER_ANGLE_THRESHOLD:
            posture_state = "Slouching"
        else:
            posture_state = "Good Posture"
            
        return posture_state, neck_angle, shoulder_angle
    
    def draw_debug_info(self, image, posture_state: str, 
                       neck_angle: float, shoulder_angle: float):
        """Draw detailed debug information on the image."""
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
        
        # Draw detailed angle information
        cv2.putText(image, f"Neck Angle: {neck_angle:.1f}° (Threshold: {self.NECK_ANGLE_THRESHOLD}°)", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.DEBUG_COLOR, 2)
        cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.1f}° (Threshold: {self.SHOULDER_ANGLE_THRESHOLD}°)", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.DEBUG_COLOR, 2)
        
        # Draw threshold status
        neck_status = "OK" if neck_angle <= self.NECK_ANGLE_THRESHOLD else "HIGH"
        shoulder_status = "OK" if shoulder_angle <= self.SHOULDER_ANGLE_THRESHOLD else "HIGH"
        
        cv2.putText(image, f"Neck Status: {neck_status}", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.GOOD_COLOR if neck_status == "OK" else self.BAD_COLOR, 2)
        cv2.putText(image, f"Shoulder Status: {shoulder_status}", 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.GOOD_COLOR if shoulder_status == "OK" else self.BAD_COLOR, 2)
        
        # Draw instructions
        cv2.putText(image, "Press 'q' to quit, 'r' to reset thresholds", 
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
        
        # Draw angle indicators
        cv2.putText(image, "N", left_ear, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "S", left_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "H", left_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def run_debug_detector(self):
        """Run the debug posture detector."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("🔍 Posture Detector Debug Mode Started!")
        print("📋 Instructions:")
        print("   - Sit at your desk")
        print("   - Watch the angle values")
        print("   - Press 'q' to quit")
        print("   - Press 'r' to reset thresholds")
        print("   - Green = Good Posture")
        print("   - Red = Slouching")
        print("   - Yellow = Debug info")
        
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
                
                # Draw debug information
                self.draw_debug_info(frame, posture_state, neck_angle, shoulder_angle)
                
                # Draw posture lines
                self.draw_posture_lines(frame, results.pose_landmarks.landmark)
            
            # Display the frame
            cv2.imshow('Posture Detector Debug', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset to even more lenient thresholds
                self.NECK_ANGLE_THRESHOLD = 90
                self.SHOULDER_ANGLE_THRESHOLD = 50
                print(f"🔄 Reset thresholds: Neck={self.NECK_ANGLE_THRESHOLD}°, Shoulder={self.SHOULDER_ANGLE_THRESHOLD}°")
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

def main():
    """Main function to run the debug posture detector."""
    detector = PostureDetectorDebug()
    detector.run_debug_detector()

if __name__ == "__main__":
    main() 