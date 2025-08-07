import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List
import threading
import queue

class PostureDetectorStreamlit:
    def __init__(self):
        """Initialize the posture detector for Streamlit."""
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
        
        # Data tracking
        self.posture_data = []
        self.current_posture = "Unknown"
        self.session_start = datetime.now()
        
        # Threading for video processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stop_thread = False
        
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate the angle between three points."""
        if not all(point1) or not all(point2) or not all(point3):
            return 0
            
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        ba = a - b
        bc = c - b
        
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
    
    def process_frame(self, frame):
        """Process a single frame and return results."""
        if frame is None:
            return None, "Unknown", 0, 0
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Analyze posture
            posture_state, neck_angle, shoulder_angle = self.analyze_posture(results.pose_landmarks.landmark)
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Draw posture lines
            self.draw_posture_lines(frame, results.pose_landmarks.landmark)
            
            return frame, posture_state, neck_angle, shoulder_angle
        
        return frame, "Unknown", 0, 0
    
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
        
        # Draw lines
        cv2.line(image, left_ear, left_shoulder, (255, 255, 0), 3)
        cv2.line(image, left_shoulder, left_hip, (255, 255, 0), 3)
        cv2.line(image, left_shoulder, right_shoulder, (0, 255, 255), 3)
    
    def video_processing_thread(self):
        """Thread for processing video frames."""
        cap = cv2.VideoCapture(0)
        
        while not self.stop_thread:
            ret, frame = cap.read()
            if ret:
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, posture_state, neck_angle, shoulder_angle = self.process_frame(frame)
                
                # Add to queues
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame)
                if not self.result_queue.full():
                    self.result_queue.put((posture_state, neck_angle, shoulder_angle))
                
                # Record posture data
                if posture_state != "Unknown":
                    self.posture_data.append({
                        'timestamp': datetime.now(),
                        'posture': posture_state,
                        'neck_angle': neck_angle,
                        'shoulder_angle': shoulder_angle
                    })
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
    
    def get_statistics(self):
        """Calculate posture statistics."""
        if not self.posture_data:
            return {
                'total_time': 0,
                'good_posture_time': 0,
                'slouching_time': 0,
                'good_posture_percentage': 0,
                'average_neck_angle': 0,
                'average_shoulder_angle': 0
            }
        
        df = pd.DataFrame(self.posture_data)
        
        # Calculate time spent in each posture
        total_time = (datetime.now() - self.session_start).total_seconds()
        good_posture_time = len(df[df['posture'] == 'Good Posture']) * 0.033  # Approximate
        slouching_time = len(df[df['posture'] == 'Slouching']) * 0.033
        
        return {
            'total_time': total_time,
            'good_posture_time': good_posture_time,
            'slouching_time': slouching_time,
            'good_posture_percentage': (good_posture_time / total_time * 100) if total_time > 0 else 0,
            'average_neck_angle': df['neck_angle'].mean(),
            'average_shoulder_angle': df['shoulder_angle'].mean()
        }

def main():
    st.set_page_config(
        page_title="Posture Detector",
        page_icon="🧍",
        layout="wide"
    )
    
    st.title("🧍 AI-Powered Posture Detector")
    st.markdown("Monitor your posture in real-time and improve your ergonomics!")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = PostureDetectorStreamlit()
        st.session_state.processing_thread = None
        st.session_state.is_running = False
    
    detector = st.session_state.detector
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    if st.sidebar.button("🚀 Start Detection", disabled=st.session_state.is_running):
        st.session_state.is_running = True
        st.session_state.processing_thread = threading.Thread(
            target=detector.video_processing_thread
        )
        st.session_state.processing_thread.start()
        st.rerun()
    
    if st.sidebar.button("⏹️ Stop Detection", disabled=not st.session_state.is_running):
        detector.stop_thread = True
        if st.session_state.processing_thread:
            st.session_state.processing_thread.join()
        st.session_state.is_running = False
        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        
        if st.session_state.is_running:
            # Create placeholder for video
            video_placeholder = st.empty()
            
            # Display video feed
            while st.session_state.is_running:
                try:
                    if not detector.frame_queue.empty():
                        frame = detector.frame_queue.get_nowait()
                        if frame is not None:
                            # Convert BGR to RGB for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    if not detector.result_queue.empty():
                        posture_state, neck_angle, shoulder_angle = detector.result_queue.get_nowait()
                        detector.current_posture = posture_state
                        
                        # Display current posture info
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Current Posture", posture_state)
                        with col_b:
                            st.metric("Neck Angle", f"{neck_angle:.1f}°")
                        with col_c:
                            st.metric("Shoulder Angle", f"{shoulder_angle:.1f}°")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    break
        else:
            st.info("Click 'Start Detection' to begin monitoring your posture.")
    
    with col2:
        st.subheader("📊 Statistics")
        
        if st.session_state.is_running:
            stats = detector.get_statistics()
            
            # Display metrics
            st.metric("Total Time", f"{stats['total_time']:.1f}s")
            st.metric("Good Posture", f"{stats['good_posture_time']:.1f}s")
            st.metric("Slouching", f"{stats['slouching_time']:.1f}s")
            st.metric("Good Posture %", f"{stats['good_posture_percentage']:.1f}%")
            
            # Create posture chart
            if detector.posture_data:
                df = pd.DataFrame(detector.posture_data)
                
                # Posture over time chart
                fig = px.line(df, x='timestamp', y='neck_angle', 
                            title="Neck Angle Over Time",
                            labels={'neck_angle': 'Neck Angle (°)', 'timestamp': 'Time'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Posture distribution pie chart
                posture_counts = df['posture'].value_counts()
                fig_pie = px.pie(values=posture_counts.values, names=posture_counts.index,
                                title="Posture Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Start detection to see statistics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Good Posture:
    - Keep your head level and aligned with your spine
    - Relax your shoulders and keep them back
    - Sit with your back straight and feet flat on the floor
    - Take regular breaks to stand and stretch
    """)

if __name__ == "__main__":
    main() 