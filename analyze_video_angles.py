#!/usr/bin/env python3
"""
Analyze video angles to help calibrate posture detection thresholds.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import Tuple, List
import argparse
import os

class VideoAngleAnalyzer:
    def __init__(self):
        """Initialize the angle analyzer."""
        self.mp_pose = mp.solutions.pose
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.angle_data = []
        
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
    
    def analyze_frame(self, landmarks) -> Tuple[float, float]:
        """Analyze angles in a frame."""
        if not landmarks:
            return 0, 0
            
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
        
        return neck_angle, shoulder_angle
    
    def analyze_video(self, video_path: str):
        """Analyze angles throughout the video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Analyzing video: {video_path}")
        print(f"📊 Video info: {total_frames} frames, {fps} FPS")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Analyze angles
                neck_angle, shoulder_angle = self.analyze_frame(results.pose_landmarks.landmark)
                
                # Store data
                self.angle_data.append({
                    'frame': frame_count,
                    'time_sec': frame_count / fps,
                    'neck_angle': neck_angle,
                    'shoulder_angle': shoulder_angle
                })
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        self.pose.close()
        
        # Analyze results
        self.print_analysis()
    
    def print_analysis(self):
        """Print detailed analysis of the angle data."""
        if not self.angle_data:
            print("❌ No pose data found in video")
            return
        
        df = pd.DataFrame(self.angle_data)
        
        print("\n📊 Angle Analysis Results")
        print("=" * 50)
        
        # Neck angle statistics
        print(f"🦴 Neck Angle Statistics:")
        print(f"   Mean: {df['neck_angle'].mean():.1f}°")
        print(f"   Median: {df['neck_angle'].median():.1f}°")
        print(f"   Min: {df['neck_angle'].min():.1f}°")
        print(f"   Max: {df['neck_angle'].max():.1f}°")
        print(f"   Std Dev: {df['neck_angle'].std():.1f}°")
        
        # Shoulder angle statistics
        print(f"\n🦴 Shoulder Angle Statistics:")
        print(f"   Mean: {df['shoulder_angle'].mean():.1f}°")
        print(f"   Median: {df['shoulder_angle'].median():.1f}°")
        print(f"   Min: {df['shoulder_angle'].min():.1f}°")
        print(f"   Max: {df['shoulder_angle'].max():.1f}°")
        print(f"   Std Dev: {df['shoulder_angle'].std():.1f}°")
        
        # Percentile analysis
        print(f"\n📈 Percentile Analysis:")
        print(f"   Neck Angle - 25th percentile: {df['neck_angle'].quantile(0.25):.1f}°")
        print(f"   Neck Angle - 75th percentile: {df['neck_angle'].quantile(0.75):.1f}°")
        print(f"   Shoulder Angle - 25th percentile: {df['shoulder_angle'].quantile(0.25):.1f}°")
        print(f"   Shoulder Angle - 75th percentile: {df['shoulder_angle'].quantile(0.75):.1f}°")
        
        # Recommended thresholds
        neck_95th = df['neck_angle'].quantile(0.95)
        shoulder_95th = df['shoulder_angle'].quantile(0.95)
        
        print(f"\n🎯 Recommended Thresholds (95th percentile):")
        print(f"   Neck Angle Threshold: {neck_95th:.1f}°")
        print(f"   Shoulder Angle Threshold: {shoulder_95th:.1f}°")
        
        # Save detailed data
        output_file = "angle_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed data saved to: {output_file}")
        
        # Show sample of data
        print(f"\n📋 Sample Data (first 5 frames):")
        print(df.head().to_string(index=False))

def main():
    """Main function to run the video angle analyzer."""
    parser = argparse.ArgumentParser(description="Analyze posture angles in a video")
    parser.add_argument("video_path", help="Path to the video file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist")
        return
    
    analyzer = VideoAngleAnalyzer()
    analyzer.analyze_video(args.video_path)

if __name__ == "__main__":
    main() 