# 🚨 Active Shooter Detection System

An advanced AI-powered system that combines **gun detection** and **pose estimation** to identify potential active shooter scenarios in real-time. The system uses YOLOv11 models for both weapon detection and human pose analysis, comparing detected poses against a database of shooting positions.

## 🎯 Features

- **Dual Detection System**: Simultaneous gun detection and pose estimation
- **Pose Database Matching**: Compares detected poses against 52 reference shooting positions
- **Real-time Processing**: Supports webcam, video files, and image analysis
- **Robust Video Output**: Multiple fallback methods ensure reliable video generation
- **Shooting Pose Alerts**: Visual and textual warnings when shooting poses are detected
- **Active Shooter Detection**: Combined gun + pose analysis for comprehensive threat assessment

## 🛠️ System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, Windows, or macOS
- **GPU**: NVIDIA GPU recommended (CUDA support) for faster processing
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: ~2GB for models and dependencies

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/putbullet/firearms-detection-system.git
cd firearms-detection-system

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Required Models

The system will automatically download YOLOv11 pose estimation models on first run. Ensure you have the gun detection model:

- Place your trained gun detection model as `weights/best.pt`
- The system supports gun, rifle, and fire detection classes

### 3. Basic Usage

#### 🖼️ Image Detection
```bash
python main.py --image path/to/image.jpg
```

#### 🎥 Video Processing
```bash
python main.py --video path/to/video.mp4
```

#### 📹 Webcam Real-time Detection
```bash
python main.py --webcam
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to image file for detection | None |
| `--video` | Path to video file for processing | None |
| `--webcam` | Use webcam for real-time detection | False |
| `--pose_threshold` | Similarity threshold for shooting pose detection (0.0-1.0) | 0.70 |

### Advanced Usage Examples

```bash
# Process video with custom pose threshold
python main.py --video shooting_test.mp4 --pose_threshold 0.65

# Real-time webcam detection with high sensitivity
python main.py --webcam --pose_threshold 0.60

# Process image with standard settings
python main.py --image security_camera.jpg
```

## 🏗️ System Architecture

### Models Used
- **Gun Detection**: Custom YOLOv11 model trained on gun/rifle/fire classes
- **Pose Estimation**: YOLOv11-pose for 17-point COCO keypoint detection
- **Pose Database**: 52 reference shooting poses from 41 images

### Processing Pipeline
1. **Input Processing**: Load image/video/webcam stream
2. **Object Detection**: Identify guns, rifles, and fire
3. **Pose Estimation**: Extract human poses (17 keypoints)
4. **Pose Analysis**: Compare against shooting position database
5. **Threat Assessment**: Combine gun + pose detections
6. **Output Generation**: Annotated results with alerts

## 📊 Performance Metrics

### Test Results
- **Processing Speed**: ~4 FPS on CPU, ~15+ FPS on GPU
- **Gun Detection Accuracy**: 85%+ on test dataset
- **Pose Matching Accuracy**: 87.6% similarity on reference poses
- **Video Processing**: Handles 1280x720 videos reliably

### Supported Formats
- **Images**: JPG, PNG, BMP
- **Videos**: MP4, AVI, MOV, MKV
- **Output**: MP4 (H.264), AVI (MJPEG) fallback

## 🗂️ Project Structure

```
firearms-detection-system/
├── main.py          # Main script
├── requirements.txt
├── README.md
├── weights/
│   ├── best.pt                        # a trained model
├── data/
│   ├── poses_database_combined.json
│   └── obj.names
├── images_vids/
│   ├── sample_images/
│   └── sample_outputs/
├── License          # MIT License

```

## 🎯 Output Examples

### Detection Results
- **Gun Detection**: Bounding boxes with confidence scores
- **Pose Analysis**: Skeleton overlay (red for shooting poses)
- **Alerts**: "SHOOTING POSE DETECTED!" text overlay
- **Active Shooter Warning**: "ACTIVE SHOOTER ALERT!" when both guns and poses detected

### Sample Output
```
✅ Enhanced detection complete! Result saved as 'output_image_detected.jpg'
⚠️ WARNING: Shooting pose detected with 1 matches!
  - Match: military_shooter12.png_pose_1 (similarity: 0.876)
```

## ⚙️ Configuration

### Pose Detection Sensitivity
- **High Sensitivity** (0.60): Detects more potential poses, may have false positives
- **Standard** (0.70): Balanced detection with good accuracy
- **High Precision** (0.80): Only very confident matches, may miss some poses

### Video Output Options
The system automatically handles video encoding with multiple fallbacks:
1. **Primary**: FFmpeg with H.264 encoding
2. **Fallback**: OpenCV with MJPEG codec
3. **Emergency**: Frame-by-frame saving + manual assembly

## 🔧 Troubleshooting

### Common Issues

#### "Model not found" Error
```bash
# Ensure the gun detection model exists
ls weights/best.pt

# If missing, train your own or obtain a pre-trained model
```

#### Video Processing Fails
```bash
# Check FFmpeg installation
ffmpeg -version

# Install FFmpeg (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg

# Install FFmpeg (macOS)
brew install ffmpeg
```

#### Out of Memory Error
```bash
# Reduce video resolution or use smaller model
# Edit the resize parameters in the code:
frame_output = cv2.resize(frame_copy, (640, 480))  # Instead of (800, 600)
```

### Performance Optimization

#### For Better Speed
- Use GPU-enabled PyTorch installation
- Reduce input resolution
- Use YOLOv11n (nano) instead of larger models
- Process every 2nd or 3rd frame for video

#### For Better Accuracy
- Use higher resolution inputs
- Lower pose threshold (0.65 instead of 0.70)
- Use YOLOv11x (extra-large) pose model
- Add more reference poses to database

## 🔒 Safety Considerations

### Important Notice
This system is designed for **security and research purposes only**. It should be used responsibly and in compliance with local laws and regulations.

### Recommended Use Cases
- **Security Monitoring**: CCTV analysis in sensitive areas
- **Research**: Academic studies on threat detection
- **Training**: Security personnel training simulations
- **Testing**: Evaluation of security systems

### Limitations
- **False Positives**: May detect innocent actions as threats
- **Lighting Dependency**: Performance varies with lighting conditions
- **Pose Variations**: Limited to trained shooting positions
- **Processing Latency**: Real-time performance depends on hardware

## 📈 Future Enhancements

### Planned Features
- [ ] Additional pose databases (tactical, hunting, sport shooting)
- [ ] Audio analysis integration
- [ ] Multiple person tracking
- [ ] Cloud deployment options
- [ ] Mobile app integration
- [ ] Real-time alerts system

### Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv11**: Ultralytics team for the excellent YOLO implementation
- **OpenCV**: Computer vision library
- **COCO Dataset**: Pose keypoint annotations
- **Research Community**: Various papers and datasets used for training

## 📞 Support

For questions, issues, or support:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the command line options

---

**⚠️ Disclaimer**: This software is provided for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.
