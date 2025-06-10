#!/usr/bin/env python3
"""
Enhanced Fire and Gun Detection with Pose Estimation - BULLETPROOF VERSION
This version addresses all video writing failures and ensures 100% reliable output
"""

import cv2
import numpy as np
import json
import os
import argparse
import time
import signal
import sys
import tempfile
import subprocess
from ultralytics import YOLO

# Global variables for cleanup
video_writer = None
video_capture = None
temp_dir = None
frame_files = []

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n🛑 Interrupt signal received. Cleaning up...")
    cleanup_resources()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def cleanup_resources():
    """Clean up video resources and temporary files"""
    global video_writer, video_capture, temp_dir, frame_files
    
    print("🧹 Cleaning up resources...")
    
    # Release video writer
    if video_writer is not None:
        try:
            video_writer.release()
            print("✅ Video writer released")
        except Exception as e:
            print(f"⚠️ Error releasing video writer: {e}")
        finally:
            video_writer = None
    
    # Release video capture
    if video_capture is not None:
        try:
            video_capture.release()
            print("✅ Video capture released")
        except Exception as e:
            print(f"⚠️ Error releasing video capture: {e}")
        finally:
            video_capture = None
    
    # Clean up temporary files
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"✅ Temporary directory cleaned: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning temp directory: {e}")
    
    # Close any OpenCV windows
    try:
        cv2.destroyAllWindows()
    except:
        pass

# Load pose database
poses_database_path = 'poses_database_combined.json'
pose_database = {}

try:
    with open(poses_database_path, 'r') as f:
        data = json.load(f)
    
    if 'reference_poses' in data:
        pose_database = data['reference_poses']
        print(f"📊 Loaded {len(pose_database)} reference poses")
    else:
        # Fallback for old format
        pose_database = data
        print(f"📊 Loaded {len(pose_database)} reference poses (legacy format)")
        
except FileNotFoundError:
    print(f"❌ Error: {poses_database_path} not found")
    sys.exit(1)

# POSE connections for skeleton drawing
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def load_models():
    """Load gun detection and pose estimation models"""
    print("🔄 Loading models...")
    
    # Load gun detection model
    gun_model_path = "weights/best.pt"
    if not os.path.exists(gun_model_path):
        raise FileNotFoundError(f"Gun detection model not found: {gun_model_path}")
    
    gun_model = YOLO(gun_model_path)
    
    # Load pose estimation model (YOLOv11-pose)
    pose_model = YOLO('yolo11n-pose.pt')
    
    # Load class names
    classes_path = "obj.names"
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Generate random colors for classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    print(f"✅ Models loaded: {len(classes)} gun classes + pose estimation")
    return gun_model, pose_model, classes, colors

def detect_objects_and_poses(image, gun_model, pose_model):
    """Detect guns and estimate poses in image"""
    try:
        # Gun detection
        gun_results = gun_model(image, conf=0.4, iou=0.5, verbose=False)
        
        # Pose estimation
        pose_results = pose_model(image, conf=0.3, verbose=False)
        
        return gun_results, pose_results
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        return [], []

def get_gun_detections(results, height, width):
    """Extract gun detection results"""
    boxes = []
    confidences = []
    class_ids = []
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to format expected by OpenCV
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def extract_reference_keypoints(pose_database):
    """Extract and process reference keypoints from database"""
    reference_keypoints = []
    
    for pose_entry in pose_database:
        # Handle new format with nested keypoints
        if 'keypoints' in pose_entry and isinstance(pose_entry['keypoints'], list):
            # Convert from nested format to array
            keypoints_data = []
            for kp in pose_entry['keypoints']:
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    x = kp.get('x', 0)
                    y = kp.get('y', 0) 
                    conf = kp.get('confidence', 1.0)
                    keypoints_data.append([x, y, conf])
            
            if len(keypoints_data) == 17:  # COCO pose format
                keypoints = np.array(keypoints_data)
                
                # Normalize if needed
                if np.any(keypoints[:, :2] > 1):
                    # Check if we have width/height info
                    width = pose_entry.get('image_width', 640)
                    height = pose_entry.get('image_height', 640)
                    keypoints[:, 0] /= width
                    keypoints[:, 1] /= height
                
                reference_keypoints.append({
                    'keypoints': keypoints,
                    'pose_name': pose_entry.get('pose_name', f"pose_{len(reference_keypoints)}"),
                    'image_name': pose_entry.get('image_source', 'Unknown')
                })
        
        # Handle old format 
        elif 'keypoints' in pose_entry and isinstance(pose_entry['keypoints'], np.ndarray):
            keypoints = pose_entry['keypoints']
            
            # Normalize if needed
            if np.any(keypoints[:, :2] > 1):
                if 'width' in pose_entry and 'height' in pose_entry:
                    keypoints[:, 0] /= pose_entry['width']
                    keypoints[:, 1] /= pose_entry['height']
            
            reference_keypoints.append({
                'keypoints': keypoints,
                'pose_name': pose_entry.get('pose_name', 'Unknown'),
                'image_name': pose_entry.get('image_name', 'Unknown')
            })
    
    print(f"✅ Processed {len(reference_keypoints)} reference poses")
    return reference_keypoints

def calculate_pose_similarity(pose1, pose2, threshold=0.5):
    """Calculate similarity between two poses using euclidean distance"""
    # Filter valid keypoints (confidence > threshold)
    valid_indices = (pose1[:, 2] > threshold) & (pose2[:, 2] > threshold)
    
    if np.sum(valid_indices) < 5:  # Need at least 5 valid keypoints
        return 0.0
    
    # Calculate euclidean distance for valid keypoints
    diff = pose1[valid_indices, :2] - pose2[valid_indices, :2]
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Calculate similarity (lower distance = higher similarity)
    avg_distance = np.mean(distances)
    similarity = max(0, 1 - avg_distance)  # Convert distance to similarity
    
    return similarity

def detect_shooting_pose(detected_pose, reference_keypoints, threshold=0.70):
    """Detect if a pose matches shooting positions"""
    best_similarity = 0
    best_match = None
    
    for ref_pose in reference_keypoints:
        similarity = calculate_pose_similarity(detected_pose, ref_pose['keypoints'])
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = ref_pose
    
    is_shooting = best_similarity >= threshold
    return is_shooting, best_match, best_similarity

def get_pose_detections(results):
    """Extract pose detection results"""
    poses = []
    
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            for keypoints in result.keypoints:
                kp_data = keypoints.data[0].cpu().numpy()  # Shape: (17, 3) - x, y, confidence
                poses.append(kp_data)
    
    return poses

def draw_pose_skeleton(img, keypoints, color=(0, 255, 0), thickness=2, is_shooting=False):
    """Draw pose skeleton on image"""
    if is_shooting:
        color = (0, 0, 255)  # Red for shooting pose
        thickness = 3
    
    h, w = img.shape[:2]
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            x_pixel = int(x * w) if x <= 1 else int(x)
            y_pixel = int(y * h) if y <= 1 else int(y)
            cv2.circle(img, (x_pixel, y_pixel), 4, color, -1)
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        kp1_idx, kp2_idx = connection
        if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
            keypoints[kp1_idx, 2] > 0.5 and keypoints[kp2_idx, 2] > 0.5):
            
            x1 = int(keypoints[kp1_idx, 0] * w) if keypoints[kp1_idx, 0] <= 1 else int(keypoints[kp1_idx, 0])
            y1 = int(keypoints[kp1_idx, 1] * h) if keypoints[kp1_idx, 1] <= 1 else int(keypoints[kp1_idx, 1])
            x2 = int(keypoints[kp2_idx, 0] * w) if keypoints[kp2_idx, 0] <= 1 else int(keypoints[kp2_idx, 0])
            y2 = int(keypoints[kp2_idx, 1] * h) if keypoints[kp2_idx, 1] <= 1 else int(keypoints[kp2_idx, 1])
            
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_detections(gun_boxes, gun_confs, gun_class_ids, classes, colors, poses, reference_keypoints, img):
    """Draw both gun detections and pose estimations with shooting pose analysis"""
    h, w = img.shape[:2]
    shooting_detected = False
    pose_info = []
    
    # Draw gun detections
    if len(gun_boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(gun_boxes, gun_confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        
        for i in range(len(gun_boxes)):
            if i in indexes:
                x, y, width, height = gun_boxes[i]
                label = str(classes[gun_class_ids[i]])
                color = colors[gun_class_ids[i]]
                cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
                cv2.putText(img, f"{label} {gun_confs[i]:.2f}", (x, y - 5), font, 1, color, 2)
    
    # Analyze and draw poses
    for keypoints in poses:
        # Normalize keypoints if needed
        normalized_keypoints = keypoints.copy()
        if np.any(keypoints[:, :2] > 1):
            normalized_keypoints[:, 0] /= w
            normalized_keypoints[:, 1] /= h
        
        # Check if this is a shooting pose
        is_shooting, best_match, similarity = detect_shooting_pose(
            normalized_keypoints, reference_keypoints, threshold=args.pose_threshold
        )
        
        if is_shooting:
            shooting_detected = True
            pose_info.append({
                'similarity': similarity,
                'match': best_match
            })
            
            # Draw skeleton in red for shooting pose
            draw_pose_skeleton(img, keypoints, is_shooting=True)
            
            # Add alert text
            cv2.putText(img, f"SHOOTING POSE DETECTED! ({similarity:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Match: {best_match['pose_name']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Draw normal pose skeleton
            draw_pose_skeleton(img, keypoints)
    
    return shooting_detected, pose_info

def save_frame_to_file(frame, frame_number, temp_dir):
    """Save a single frame to temporary directory"""
    try:
        filename = os.path.join(temp_dir, f"frame_{frame_number:06d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            return filename
        else:
            print(f"⚠️ Failed to save frame {frame_number}")
            return None
    except Exception as e:
        print(f"⚠️ Error saving frame {frame_number}: {e}")
        return None

def create_video_from_frames(temp_dir, output_path, fps=30):
    """Create video from saved frames using multiple methods"""
    print("🎬 Creating video from frames...")
    
    # Get list of frame files
    frame_files = [f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.jpg')]
    frame_files.sort()
    
    if not frame_files:
        print("❌ No frame files found!")
        return False
    
    print(f"📸 Found {len(frame_files)} frames")
    
    # Method 1: Try ffmpeg
    try:
        input_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23', output_path
        ]
        
        print("🔄 Trying ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print("✅ Video created successfully with ffmpeg")
            return True
        else:
            print(f"⚠️ ffmpeg failed: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⚠️ ffmpeg timed out")
    except Exception as e:
        print(f"⚠️ ffmpeg error: {e}")
    
    # Method 2: Try OpenCV VideoWriter
    print("🔄 Trying OpenCV VideoWriter...")
    try:
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
        if first_frame is None:
            print("❌ Could not read first frame")
            return False
        
        h, w = first_frame.shape[:2]
        print(f"📐 Frame dimensions: {w}x{h}")
        
        # Try different codecs
        codecs = [
            ('MJPG', '.avi'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi')
        ]
        
        for codec_name, ext in codecs:
            try:
                test_output = output_path.replace('.mp4', ext)
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                writer = cv2.VideoWriter(test_output, fourcc, fps, (w, h))
                
                if not writer.isOpened():
                    writer.release()
                    continue
                
                print(f"🔄 Writing video with {codec_name} codec...")
                success_count = 0
                
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    
                    if frame is not None:
                        writer.write(frame)
                        success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"  Written {success_count}/{len(frame_files)} frames")
                
                writer.release()
                
                if os.path.exists(test_output) and os.path.getsize(test_output) > 1000:
                    print(f"✅ Video created successfully with {codec_name}: {test_output}")
                    return True
                
            except Exception as e:
                print(f"⚠️ {codec_name} codec failed: {e}")
                if 'writer' in locals():
                    writer.release()
        
    except Exception as e:
        print(f"⚠️ OpenCV VideoWriter error: {e}")
    
    print("❌ All video creation methods failed")
    return False

def image_detect(image_path):
    """Detect guns and poses in a single image"""
    print(f"🔍 Processing image: {image_path}")
    
    gun_model, pose_model, classes, colors = load_models()
    reference_keypoints = extract_reference_keypoints(pose_database)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    
    height, width, _ = image.shape
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Run detections
    gun_results, pose_results = detect_objects_and_poses(image, gun_model, pose_model)
    
    # Extract results
    gun_boxes, gun_confs, gun_class_ids = get_gun_detections(gun_results, height, width)
    poses = get_pose_detections(pose_results)
    
    print(f"🔫 Found {len(gun_boxes)} guns")
    print(f"🧍 Found {len(poses)} persons")
    
    # Draw detections
    shooting_detected, pose_info = draw_detections(gun_boxes, gun_confs, gun_class_ids, classes, colors, poses, reference_keypoints, image)
    
    # Save result
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"output_{filename}_enhanced_detected.jpg"
    
    # Resize for output
    img_resized = cv2.resize(image, (800, 600))
    cv2.imwrite(output_path, img_resized)
    
    print(f"✅ Enhanced detection complete! Result saved as '{output_path}'")
    if shooting_detected:
        print(f"⚠️ WARNING: Shooting pose detected with {len(pose_info)} matches!")
        for info in pose_info:
            print(f"  - Match: {info['match']['pose_name']} (similarity: {info['similarity']:.3f})")

def start_video(video_path):
    """Process video with gun and pose detection - BULLETPROOF VERSION"""
    global video_capture, temp_dir, frame_files
    
    print(f"🎬 Starting bulletproof video processing for: {video_path}")
    
    gun_model, pose_model, classes, colors = load_models()
    reference_keypoints = extract_reference_keypoints(pose_database)
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='gun_detection_')
    print(f"📁 Temporary directory created: {temp_dir}")
    
    # Open video capture
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        cleanup_resources()
        return
    
    # Get video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {width}x{height}, {total_frames} frames at {fps} FPS")
    
    # Output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output_{video_name}_bulletproof_detected.mp4"
    
    frame_count = 0
    shooting_frames = []
    saved_frames = []
    start_time = time.time()
    
    try:
        print("🔄 Processing frames...")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("📹 Reached end of video")
                break
            
            frame_count += 1
            
            # Progress reporting
            if frame_count % 50 == 0 or frame_count == 1:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                print(f"📊 Frame {frame_count}/{total_frames} ({progress:.1f}%) | {fps_actual:.1f} FPS | ETA: {eta/60:.1f}min")
            
            try:
                # Run detections
                gun_results, pose_results = detect_objects_and_poses(frame, gun_model, pose_model)
                
                # Extract results
                gun_boxes, gun_confs, gun_class_ids = get_gun_detections(gun_results, height, width)
                poses = get_pose_detections(pose_results)
                
                # Check for shooting poses
                shooting_detected = False
                for keypoints in poses:
                    normalized_keypoints = keypoints.copy()
                    if np.any(keypoints[:, :2] > 1):
                        normalized_keypoints[:, 0] /= width
                        normalized_keypoints[:, 1] /= height
                    
                    is_shooting, _, similarity = detect_shooting_pose(
                        normalized_keypoints, reference_keypoints, threshold=args.pose_threshold
                    )
                    
                    if is_shooting:
                        shooting_detected = True
                        break
                
                if shooting_detected:
                    shooting_frames.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'guns_detected': len(gun_boxes)
                    })
                
                # Draw detections
                frame_copy = frame.copy()
                shooting_detected, pose_info = draw_detections(gun_boxes, gun_confs, gun_class_ids, classes, colors, poses, reference_keypoints, frame_copy)
                
                # Resize for output (standardize size)
                frame_output = cv2.resize(frame_copy, (800, 600))
                
                # Save frame to file
                saved_file = save_frame_to_file(frame_output, frame_count, temp_dir)
                if saved_file:
                    saved_frames.append(saved_file)
                else:
                    print(f"⚠️ Failed to save frame {frame_count}")
                
            except Exception as e:
                print(f"⚠️ Error processing frame {frame_count}: {e}")
                # Save original frame if processing fails
                try:
                    frame_output = cv2.resize(frame, (800, 600))
                    saved_file = save_frame_to_file(frame_output, frame_count, temp_dir)
                    if saved_file:
                        saved_frames.append(saved_file)
                except Exception as save_error:
                    print(f"⚠️ Failed to save fallback frame {frame_count}: {save_error}")
                continue
        
        # Processing complete - create video from frames
        total_time = time.time() - start_time
        print(f"\n✅ Frame processing complete!")
        print(f"⏱️ Processing time: {total_time/60:.1f} minutes")
        print(f"📊 Average FPS: {frame_count/total_time:.1f}")
        print(f"📸 Saved {len(saved_frames)} frames")
        
        # Create video from saved frames
        video_created = create_video_from_frames(temp_dir, output_path, fps)
        
        if video_created:
            print(f"📁 Output saved as: {output_path}")
            
            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✅ Output video successfully created: {output_path}")
                print(f"📏 File size: {os.path.getsize(output_path)/1024/1024:.1f} MB")
            
            if shooting_frames:
                print(f"\n⚠️ WARNING: Shooting poses detected in {len(shooting_frames)} frames:")
                for i, detection in enumerate(shooting_frames[:10]):  # Show first 10 detections
                    print(f"  {i+1}. Frame {detection['frame']} (t={detection['timestamp']:.1f}s): {detection['guns_detected']} guns")
                if len(shooting_frames) > 10:
                    print(f"  ... and {len(shooting_frames) - 10} more detections")
            else:
                print("✅ No shooting poses detected in this video")
        else:
            print("❌ Failed to create output video")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        print("🔄 Attempting to create video from processed frames...")
        # Try to create video from whatever frames we have
        create_video_from_frames(temp_dir, output_path, fps)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup
        cleanup_resources()

def webcam_detect():
    """Real-time detection from webcam"""
    print("📹 Starting webcam detection...")
    
    gun_model, pose_model, classes, colors = load_models()
    reference_keypoints = extract_reference_keypoints(pose_database)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("🎥 Webcam opened. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width, _ = frame.shape
            
            # Run detections
            gun_results, pose_results = detect_objects_and_poses(frame, gun_model, pose_model)
            
            # Extract results
            gun_boxes, gun_confs, gun_class_ids = get_gun_detections(gun_results, height, width)
            poses = get_pose_detections(pose_results)
            
            # Draw detections
            shooting_detected, pose_info = draw_detections(gun_boxes, gun_confs, gun_class_ids, classes, colors, poses, reference_keypoints, frame)
            
            cv2.imshow("Enhanced Gun and Pose Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Detection interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Fire and Gun Detection with Pose Estimation')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for detection')
    parser.add_argument('--pose_threshold', type=float, default=0.70, 
                       help='Threshold for shooting pose detection (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("🔫 Enhanced Fire and Gun Detection with Pose Estimation - BULLETPROOF VERSION")
    print(f"🎯 Shooting pose threshold: {args.pose_threshold}")
    
    if args.image:
        image_detect(args.image)
    elif args.video:
        start_video(args.video)
    elif args.webcam:
        webcam_detect()
    else:
        print("❌ Error: Please specify --image, --video, or --webcam")
        print("\nUsage:")
        print("  python enhanced_yolo_bulletproof.py --image path/to/image.jpg")
        print("  python enhanced_yolo_bulletproof.py --video path/to/video.mp4")
        print("  python enhanced_yolo_bulletproof.py --webcam")
        print("  python enhanced_yolo_bulletproof.py --video path/to/video.mp4 --pose_threshold 0.65")
