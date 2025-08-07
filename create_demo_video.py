#!/usr/bin/env python3
"""
Create a demo video for testing the posture detector.
This generates a simple video showing different postures.
"""

import cv2
import numpy as np
import os

def create_demo_video(output_path="demo_video.mp4", duration=10, fps=30):
    """Create a demo video with posture changes."""
    
    # Video settings
    width, height = 640, 480
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎬 Creating demo video: {output_path}")
    print(f"📊 Settings: {width}x{height}, {fps} FPS, {duration} seconds")
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(height):
            color = int(255 * (1 - y / height))
            frame[y, :] = [color, color, color]
        
        # Calculate time and posture phase
        time_sec = frame_num / fps
        phase = (time_sec / 2) % 4  # 4-second cycles
        
        # Draw different postures based on phase
        if phase < 1:  # Good posture
            posture_text = "Good Posture"
            color = (0, 255, 0)  # Green
            # Draw upright stick figure
            cv2.line(frame, (320, 100), (320, 300), (255, 255, 255), 3)  # Spine
            cv2.line(frame, (320, 150), (280, 200), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (320, 150), (360, 200), (255, 255, 255), 3)  # Right arm
            cv2.line(frame, (320, 300), (280, 400), (255, 255, 255), 3)  # Left leg
            cv2.line(frame, (320, 300), (360, 400), (255, 255, 255), 3)  # Right leg
            cv2.circle(frame, (320, 80), 20, (255, 255, 255), -1)  # Head
            
        elif phase < 2:  # Slouching posture
            posture_text = "Slouching"
            color = (0, 0, 255)  # Red
            # Draw slouched stick figure
            cv2.line(frame, (320, 100), (340, 300), (255, 255, 255), 3)  # Spine (tilted)
            cv2.line(frame, (340, 150), (300, 220), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (340, 150), (380, 220), (255, 255, 255), 3)  # Right arm
            cv2.line(frame, (340, 300), (300, 400), (255, 255, 255), 3)  # Left leg
            cv2.line(frame, (340, 300), (380, 400), (255, 255, 255), 3)  # Right leg
            cv2.circle(frame, (330, 80), 20, (255, 255, 255), -1)  # Head (tilted)
            
        elif phase < 3:  # Transition back to good
            posture_text = "Improving Posture"
            color = (0, 165, 255)  # Orange
            # Draw intermediate stick figure
            cv2.line(frame, (320, 100), (325, 300), (255, 255, 255), 3)  # Spine
            cv2.line(frame, (325, 150), (290, 210), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (325, 150), (360, 210), (255, 255, 255), 3)  # Right arm
            cv2.line(frame, (325, 300), (290, 400), (255, 255, 255), 3)  # Left leg
            cv2.line(frame, (325, 300), (360, 400), (255, 255, 255), 3)  # Right leg
            cv2.circle(frame, (322, 80), 20, (255, 255, 255), -1)  # Head
            
        else:  # Good posture again
            posture_text = "Good Posture"
            color = (0, 255, 0)  # Green
            # Draw upright stick figure
            cv2.line(frame, (320, 100), (320, 300), (255, 255, 255), 3)  # Spine
            cv2.line(frame, (320, 150), (280, 200), (255, 255, 255), 3)  # Left arm
            cv2.line(frame, (320, 150), (360, 200), (255, 255, 255), 3)  # Right arm
            cv2.line(frame, (320, 300), (280, 400), (255, 255, 255), 3)  # Left leg
            cv2.line(frame, (320, 300), (360, 400), (255, 255, 255), 3)  # Right leg
            cv2.circle(frame, (320, 80), 20, (255, 255, 255), -1)  # Head
        
        # Add text
        cv2.putText(frame, posture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Time: {time_sec:.1f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Demo Video for Posture Detector", (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add angle indicators
        if "Good" in posture_text:
            cv2.putText(frame, "Neck Angle: 25°", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Shoulder Angle: 5°", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif "Slouching" in posture_text:
            cv2.putText(frame, "Neck Angle: 55°", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Shoulder Angle: 25°", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Neck Angle: 35°", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Shoulder Angle: 15°", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📈 Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # Cleanup
    out.release()
    print(f"✅ Demo video created: {output_path}")
    print(f"📊 Video contains {total_frames} frames ({duration} seconds)")

def main():
    """Main function to create demo video."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a demo video for posture detector testing")
    parser.add_argument("--output", "-o", default="demo_video.mp4", help="Output video file path")
    parser.add_argument("--duration", "-d", type=int, default=10, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    create_demo_video(args.output, args.duration, args.fps)

if __name__ == "__main__":
    main() 