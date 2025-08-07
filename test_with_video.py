#!/usr/bin/env python3
"""
Test script for Posture Detector that works with video files or images.
Useful for testing on systems without webcams (like Mac mini).
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import os
from typing import Tuple

class PostureDetectorVideo:
    def __init__(self):
        """Initialize the posture detector for video files."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Posture thresholds (optimized based on all 4 videos analysis)
        self.NECK_ANGLE_THRESHOLD = 132  # degrees (balanced average from 3 videos)
        self.SHOULDER_ANGLE_THRESHOLD = 3  # degrees (balanced average from 3 videos)
        
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
    
    def process_video(self, input_path: str, output_path: str = None, show_preview: bool = True):
        """Process a video file and analyze posture."""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {input_path}")
        print(f"📊 Video info: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"💾 Output will be saved to: {output_path}")
        
        frame_count = 0
        posture_stats = {"Good Posture": 0, "Slouching": 0, "Unknown": 0}
        
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
                # Analyze posture
                posture_state, neck_angle, shoulder_angle = self.analyze_posture(results.pose_landmarks.landmark)
                posture_stats[posture_state] += 1
                
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
            
            # Write frame to output video
            if video_writer:
                video_writer.write(frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Posture Detector - Video Processing', frame)
                
                # Press 'q' to quit preview
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress indicator
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n📊 Processing Complete!")
        print("=" * 50)
        total_processed = sum(posture_stats.values())
        for posture, count in posture_stats.items():
            percentage = (count / total_processed * 100) if total_processed > 0 else 0
            print(f"{posture}: {count} frames ({percentage:.1f}%)")
        
        self.pose.close()
    
    def process_image(self, input_path: str, output_path: str = None, show_preview: bool = True):
        """Process a single image and analyze posture."""
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Error: Could not load image {input_path}")
            return
        
        print(f"🖼️  Processing image: {input_path}")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Analyze posture
            posture_state, neck_angle, shoulder_angle = self.analyze_posture(results.pose_landmarks.landmark)
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Draw posture feedback
            self.draw_posture_feedback(image, posture_state, neck_angle, shoulder_angle)
            
            # Draw posture lines
            self.draw_posture_lines(image, results.pose_landmarks.landmark)
            
            print(f"📊 Analysis Results:")
            print(f"   Posture: {posture_state}")
            print(f"   Neck Angle: {neck_angle:.1f}°")
            print(f"   Shoulder Angle: {shoulder_angle:.1f}°")
        else:
            print("❌ No pose detected in the image")
        
        # Save output image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"💾 Output saved to: {output_path}")
        
        # Show preview
        if show_preview:
            cv2.imshow('Posture Detector - Image Processing', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        self.pose.close()

def main():
    """Main function to run the video/image processor."""
    parser = argparse.ArgumentParser(description="Process videos or images for posture detection")
    parser.add_argument("input", help="Input video file or image path")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--image", action="store_true", help="Force image processing mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    detector = PostureDetectorVideo()
    
    # Determine if input is image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = os.path.splitext(args.input)[1].lower()
    
    if args.image or file_ext in image_extensions:
        detector.process_image(args.input, args.output, not args.no_preview)
    else:
        detector.process_video(args.input, args.output, not args.no_preview)

if __name__ == "__main__":
    main() 