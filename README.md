# 🧍 AI-Powered Posture Detector

A real-time pose estimation tool that detects poor posture using computer vision to help improve ergonomics at your desk.

## 🎯 Overview

This project uses MediaPipe pose estimation to analyze your posture in real-time, providing instant feedback on whether you're maintaining good posture or slouching. Perfect for remote workers, students, or anyone who spends long hours at a desk.

<img width="1993" height="1047" alt="Screenshot 2025-08-22 at 4 56 51 PM" src="https://github.com/user-attachments/assets/44502a40-3ee8-4124-bc7b-e831ec10da4d" />


<img width="1830" height="1026" alt="Screenshot 2025-08-22 at 4 56 34 PM" src="https://github.com/user-attachments/assets/1637ce59-f47f-486f-a93e-e940bee2b50a" />




## ✨ Features![Uploading Screenshot 2025-08-22 at 4.56.34 PM.png…]()


- **Real-time Pose Detection**: Uses MediaPipe to detect 33 body landmarks
- **Posture Analysis**: Calculates neck and shoulder angles to determine posture quality
- **Visual Feedback**: Color-coded feedback (Green = Good, Red = Slouching)
- **Live Statistics**: Track your posture duration and percentages
- **Two Interfaces**: 
  - Command-line interface (`main.py`)
  - Web dashboard (`streamlit_app.py`)
- **Posture Lines**: Visual overlay showing neck and spine alignment
- **Session Tracking**: Monitor your posture habits over time

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Pose Estimation | MediaPipe |
| Computer Vision | OpenCV |
| Web Interface | Streamlit |
| Data Visualization | Plotly |
| Language | Python 3.8+ |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone [<repository-url>](https://github.com/ghaly-jd/Pose-estimation-for-Ergonomics)
   cd Pose-estimation-for-Ergonomics
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, mediapipe, streamlit; print('✅ All dependencies installed!')"
   ```

## 🚀 Usage

### Option 1: Command Line Interface

Run the basic posture detector:
```bash
python main.py
```

**Controls:**
- Press `q` to quit
- Sit at your desk facing the camera
- Watch for color-coded feedback

### Option 2: Streamlit Dashboard

Run the enhanced web interface:
```bash
streamlit run streamlit_app.py
```

**Features:**
- Start/Stop detection with buttons
- Real-time statistics and charts
- Posture duration tracking
- Interactive visualizations

## 📊 How It Works

### Posture Classification Logic

**Good Posture Criteria:**
- Neck angle < 45° (head not tilted too far forward)
- Shoulder alignment < 15° (shoulders level)
- Back relatively straight

**Poor Posture Indicators:**
- Head tilting forward (neck angle > 45°)
- Shoulders hunched or uneven
- Spine not aligned

### Key Measurements

1. **Neck Angle**: Calculated from ear → shoulder → hip
2. **Shoulder Angle**: Horizontal alignment of shoulders
3. **Posture Duration**: Time spent in each posture state

## 🎥 Demo

To create a demo video:

1. **Record your session** using screen recording software
2. **Show the interface** with live posture detection
3. **Demonstrate posture changes** (sit straight, then slouch)
4. **Highlight the feedback** (color changes, angle measurements)

## 📈 Features Breakdown

### ✅ Completed
- [x] Live webcam pose detection
- [x] Calculate key angles (neck, shoulder)
- [x] Classify posture (upright vs slouching)
- [x] Display real-time feedback
- [x] Visual posture lines overlay
- [x] Session statistics tracking
- [x] Dual interface options

### 🔄 Stretch Goals (Future)
- [ ] Alert sounds after prolonged bad posture
- [ ] CSV export of posture logs
- [ ] Multiple camera support
- [ ] Mobile app version
- [ ] Machine learning model training

## 🎛️ Configuration

### Adjustable Parameters

In `main.py` and `streamlit_app.py`, you can modify:

```python
# Posture thresholds
self.NECK_ANGLE_THRESHOLD = 45  # degrees
self.SHOULDER_ANGLE_THRESHOLD = 15  # degrees

# Detection confidence
min_detection_confidence=0.5
min_tracking_confidence=0.5
```

### Customization Tips

- **Lower thresholds** = More sensitive detection
- **Higher thresholds** = Less sensitive, fewer false positives
- **Adjust confidence** = Balance between accuracy and performance

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not found**
   ```bash
   # Check available cameras
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```

2. **Performance issues**
   - Reduce frame resolution
   - Lower detection confidence
   - Close other applications

3. **Dependencies not found**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

### System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **Camera**: Built-in webcam or USB camera
- **RAM**: 4GB+ recommended
- **GPU**: Optional (CPU works fine)

## 📁 Project Structure

```
Pose-estimation-for-Ergonomics/
├── main.py              # Command-line posture detector
├── streamlit_app.py     # Web dashboard interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **MediaPipe** for pose estimation capabilities
- **OpenCV** for computer vision tools
- **Streamlit** for the web interface framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues
3. Create a new issue with details about your problem

---

**Happy Posture Monitoring! 🧍‍♂️✨** 
