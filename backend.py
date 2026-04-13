import os
import sys

# Fix for PyTorch DLL issue on Windows
torch_lib = os.path.join(os.path.dirname(__file__), 'venv', 'Lib', 'site-packages', 'torch', 'lib')
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import csv
import onnxruntime as ort
from enum import Enum
from PySide6.QtCore import QObject, Signal, Slot, Property, QTimer, Qt, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtMultimedia import QMediaDevices, QCamera, QMediaCaptureSession, QVideoSink, QImageCapture, QVideoFrame 
from PySide6.QtCore import QThread, Signal # recently added


class EmotionDetectionMethod(Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"

class EmotionDetectorBackend(QObject):
    # Signals
    frame_ready = Signal(QImage)
    emotion_detected = Signal(str, float, dict)
    face_count_updated = Signal(int)
    fps_updated = Signal(float)
    session_stats_updated = Signal(dict)
    error_occurred = Signal(str)
    available_cameras_changed = Signal(list)
    detection_method_changed = Signal(str)
    model_status_changed = Signal(str, bool)

    # Properties
    is_camera_active_changed = Signal(bool)
    is_recording_changed = Signal(bool)
    model_loaded_changed = Signal(bool)
    confidence_threshold_changed = Signal(float)

    # Emotion labels (exactly from training script)
    EMOTION_LABELS = ['happy', 'sad']

    def __init__(self):
        super().__init__()

        # Camera
        self._camera = None
        self._capture_session = None
        self._video_sink = None
        self._image_capture = None
        self._current_camera_index = 0
        self._is_camera_active = False
        
        # Frame processing thread

        self._frame_processor = FrameProcessor(self)

        # Connect thread signals
        self._frame_processor.frame_ready.connect(self.frame_ready)
        self._frame_processor.emotion_detected.connect(self.emotion_detected)
        self._frame_processor.face_count_updated.connect(self.face_count_updated)

        # Start thread
        self._frame_processor.start()

        # Models
        self._pytorch_model = None
        self._onnx_session = None
        self._torchscript_model = None
        self._model_loaded = False
        self._confidence_threshold = 0.5

        # Detection method
        self._detection_method = EmotionDetectionMethod.PYTORCH
        self._available_methods = []

        # Face detection
        self._face_cascade = None
        self._face_detection_method = "haar"
        self._face_dnn_net = None

        # Model normalization parameters (from training script)
        self._model_mean = 0.5
        self._model_std = 0.5
        self._input_size = (48, 48)

        # Analytics
        self._is_recording = False
        self._detections = []
        self._session_stats = {
            'total_detections': 0,
            'emotion_distribution': {},
            'average_confidence': 0.0,
            'start_time': '',
            'end_time': ''
        }

        # Performance
        self._frame_counter = 0
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps)
        self._frame_times = []

        # Initialize
        self._init_face_cascade()
        self._init_face_dnn()
        self._load_all_models()

    # Helper method to convert QImage to numpy array correctly
    def _qimage_to_numpy(self, qimage):
        """Convert QImage to numpy array correctly"""
        try:
            # Convert to RGB888 format first
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            width = qimage.width()
            height = qimage.height()

            # Get bytes
            ptr = qimage.constBits()
            if ptr is None:
                return None

            # Copy to numpy array
            arr = np.frombuffer(ptr, dtype=np.uint8, count=width * height * 3)
            arr = arr.reshape((height, width, 3))

            return arr

        except Exception as e:
            print(f"Error converting QImage to numpy: {str(e)}")
            return None

    # Properties
    @Property(bool, notify=is_camera_active_changed)
    def is_camera_active(self):
        return self._is_camera_active

    @is_camera_active.setter
    def is_camera_active(self, value):
        if self._is_camera_active != value:
            self._is_camera_active = value
            self.is_camera_active_changed.emit(value)

    @Property(bool, notify=is_recording_changed)
    def is_recording(self):
        return self._is_recording

    @is_recording.setter
    def is_recording(self, value):
        if self._is_recording != value:
            self._is_recording = value
            if value:
                self._start_session()
            else:
                self._end_session()
            self.is_recording_changed.emit(value)

    @Property(float, notify=confidence_threshold_changed)
    def confidence_threshold(self):
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value):
        if self._confidence_threshold != value:
            self._confidence_threshold = max(0.1, min(0.99, value))
            self.confidence_threshold_changed.emit(self._confidence_threshold)

    @Property(bool, notify=model_loaded_changed)
    def model_loaded(self):
        return self._model_loaded

    @Property(list, notify=detection_method_changed)
    def available_methods(self):
        return self._available_methods

    @Slot()
    def initialize(self):
        """Initialize the application"""
        self._update_available_cameras()
        self.model_status_changed.emit("Initialized", True)

    @Slot()
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self._is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    @Slot()
    def start_camera(self):
        """Start camera capture"""
        try:
            devices = QMediaDevices.videoInputs()
            if not devices:
                self.error_occurred.emit("No camera found")
                return

            device = devices[self._current_camera_index % len(devices)]
            self._camera = QCamera(device)
            self._capture_session = QMediaCaptureSession()
            self._video_sink = QVideoSink()

            # Configure camera
            self._capture_session.setCamera(self._camera)
            self._capture_session.setVideoSink(self._video_sink)

            # Connect video frame signal
            self._video_sink.videoFrameChanged.connect(self._frame_processor.submit_frame)

            # Start camera
            self._camera.start()
            self._is_camera_active = True
            self.is_camera_active_changed.emit(True)

            # Start FPS timer
            self._fps_timer.start(1000)

        except Exception as e:
            self.error_occurred.emit(f"Failed to start camera: {str(e)}")

    @Slot()
    def stop_camera(self):
        """Stop camera capture"""
        try:
            if self._fps_timer.isActive():
                self._fps_timer.stop()

            if self._camera:
                self._camera.stop()
                self._camera = None

            self._capture_session = None
            self._video_sink = None
            self._frame_counter = 0
            self._is_camera_active = False
            self.is_camera_active_changed.emit(False)

            # Send a blank frame when camera is stopped
            blank_frame = QImage(640, 480, QImage.Format_RGB888)
            blank_frame.fill(Qt.gray)
            self.frame_ready.emit(blank_frame)

        except Exception as e:
            self.error_occurred.emit(f"Failed to stop camera: {str(e)}")

    @Slot(int)
    def set_camera(self, index):
        """Change camera device"""
        if index != self._current_camera_index:
            self._current_camera_index = index
            if self._is_camera_active:
                self.stop_camera()
                QTimer.singleShot(100, self.start_camera)

    @Slot(str)
    def set_detection_method(self, method_str):
        """Set emotion detection method"""
        try:
            method = EmotionDetectionMethod(method_str.lower())
            if method != self._detection_method:
                self._detection_method = method
                self.detection_method_changed.emit(method.value)
                self.model_status_changed.emit(f"Switched to {method.value.capitalize()}", True)
        except ValueError:
            self.error_occurred.emit(f"Invalid detection method: {method_str}")

    @Slot()
    def toggle_recording(self):
        """Toggle recording session"""
        self.is_recording = not self.is_recording

    @Slot(result=str)
    def capture_frame(self):
        """Capture current frame as image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"captures/capture_{timestamp}.png"

            Path("captures").mkdir(exist_ok=True)

            image = QImage(640, 480, QImage.Format_RGB32)
            image.fill(Qt.gray)

            if image.save(filename):
                return filename
        except Exception as e:
            print(f"Capture error: {str(e)}")
        return ""

    @Slot(str, result=bool)
    def export_session_data(self, filename):
        """Export session data to CSV"""
        try:
            if not self._detections:
                return False

            Path("exports").mkdir(exist_ok=True)

            if not filename.lower().endswith('.csv'):
                filename += '.csv'

            filepath = Path("exports") / filename

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'emotion', 'confidence', 'method'])
                for detection in self._detections:
                    writer.writerow([
                        detection['timestamp'],
                        detection['emotion'],
                        f"{detection['confidence']:.4f}",
                        detection.get('method', 'unknown')
                    ])
            return True
        except Exception as e:
            print(f"Export error: {str(e)}")
            return False

    @Slot()
    def clear_session(self):
        """Clear current session data"""
        self._detections.clear()
        self._session_stats = {
            'total_detections': 0,
            'emotion_distribution': {},
            'average_confidence': 0.0,
            'start_time': '',
            'end_time': ''
        }
        self.session_stats_updated.emit(self._session_stats)

    def _preprocess_face_for_model(self, face_image):
        """Preprocess face image for your specific models (48x48 grayscale)"""
        try:
            # Convert QImage to numpy
            arr = self._qimage_to_numpy(face_image)
            if arr is None:
                return None

            # Convert RGB to BGR (OpenCV format)
            bgr_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Resize to 48x48
            resized = cv2.resize(bgr_arr, self._input_size)

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Normalize: [0, 255] -> [0, 1] -> apply (x - mean)/std
            normalized = gray.astype(np.float32) / 255.0
            normalized = (normalized - self._model_mean) / self._model_std

            return normalized

        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None

    def _process_frame(self, frame: QImage):
        if frame is None or frame.isNull():  # ✅ correct QImage check
            return

        try:
            import time
            start_time = time.time()

            self._frame_counter += 1

            # No need to convert frame to QImage again
            # frame is already a QImage
            display_image = frame.convertToFormat(QImage.Format_RGB888)

            # Create a working copy for processing
            working_image = display_image.copy()

            # Detect faces
            faces = self._detect_faces(working_image)
            self.face_count_updated.emit(len(faces))

            # Process faces for emotion detection
            if faces and self._model_loaded:
                for (x, y, w, h) in faces:
                    face_rect = QRect(x, y, w, h)
                    face_img = working_image.copy(face_rect)

                    emotion, confidence, probabilities = self._detect_emotion(face_img)

                    if confidence >= self._confidence_threshold:
                        self._draw_on_image(display_image, face_rect, emotion, confidence)
                        self.emotion_detected.emit(emotion, confidence, probabilities)

                        if self._is_recording:
                            self._record_detection(emotion, confidence, probabilities)

            # Emit the final image for GUI display
            self.frame_ready.emit(display_image)

            # Track processing time
            processing_time = time.time() - start_time
            self._frame_times.append(processing_time)
            if len(self._frame_times) > 10:
                self._frame_times.pop(0)

        except Exception as e:
            print(f"Frame processing error: {str(e)}")

    def _detect_faces(self, image):
        """Detect faces in image using OpenCV"""
        faces = []

        try:
            width = image.width()
            height = image.height()

            if width == 0 or height == 0:
                return faces

            # Convert QImage to numpy
            arr = self._qimage_to_numpy(image)
            if arr is None:
                return faces

            # Convert RGB to BGR for OpenCV
            bgr_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            if self._face_detection_method == "dnn" and self._face_dnn_net is not None:
                h, w = bgr_arr.shape[:2]
                blob = cv2.dnn.blobFromImage(bgr_arr, 1.0, (300, 300),
                                            [104, 117, 123], False, False)

                self._face_dnn_net.setInput(blob)
                detections = self._face_dnn_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x2, y2) = box.astype("int")
                        w_face = x2 - x
                        h_face = y2 - y
                        faces.append((x, y, w_face, h_face))

            else:
                gray = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2GRAY)

                if self._face_cascade is not None and not self._face_cascade.empty():
                    detected = self._face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    for (x, y, w, h) in detected:
                        faces.append((x, y, w, h))

        except Exception as e:
            print(f"Face detection error: {str(e)}")

        return faces

    def _detect_emotion(self, face_image):
        """Detect emotion from face image using selected method"""
        emotion = "neutral"
        confidence = 0.0
        probabilities = {label: 0.0 for label in self.EMOTION_LABELS}

        try:
            if not self._model_loaded:
                return self._get_mock_emotion()

            # Preprocess the face image (48x48 grayscale, normalized)
            processed = self._preprocess_face_for_model(face_image)
            if processed is None:
                return emotion, confidence, probabilities

            if self._detection_method == EmotionDetectionMethod.PYTORCH and self._pytorch_model:
                emotion, confidence, probabilities = self._detect_emotion_pytorch(processed)

            elif self._detection_method == EmotionDetectionMethod.ONNX and self._onnx_session:
                emotion, confidence, probabilities = self._detect_emotion_onnx(processed)

            elif self._detection_method == EmotionDetectionMethod.TORCHSCRIPT and self._torchscript_model:
                emotion, confidence, probabilities = self._detect_emotion_torchscript(processed)

            else:
                return self._get_mock_emotion()

        except Exception as e:
            print(f"Emotion detection error ({self._detection_method.value}): {str(e)}")
            return self._get_mock_emotion()

        return emotion, confidence, probabilities

    def _detect_emotion_pytorch(self, processed_face):
        """Detect emotion using PyTorch model"""
        emotion = "neutral"
        confidence = 0.0
        probabilities = {label: 0.0 for label in self.EMOTION_LABELS}

        try:
            if self._pytorch_model is None:
                return emotion, confidence, probabilities

            # Convert to tensor: [1, 1, 48, 48]
            tensor = torch.FloatTensor(processed_face).unsqueeze(0).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = self._pytorch_model(tensor)

                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # Get results
                confidence, predicted_idx = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()

                if predicted_idx < len(self.EMOTION_LABELS):
                    emotion = self.EMOTION_LABELS[predicted_idx]
                    for i, label in enumerate(self.EMOTION_LABELS):
                        probabilities[label] = probs[0][i].item()

        except Exception as e:
            print(f"PyTorch emotion detection error: {str(e)}")

        return emotion, confidence, probabilities

    def _detect_emotion_onnx(self, processed_face):
        """Detect emotion using ONNX model"""
        emotion = "neutral"
        confidence = 0.0
        probabilities = {label: 0.0 for label in self.EMOTION_LABELS}

        try:
            if self._onnx_session is None:
                return emotion, confidence, probabilities

            # Prepare input for ONNX: [1, 1, 48, 48]
            input_tensor = processed_face.reshape(1, 1, 48, 48).astype(np.float32)

            # Run inference
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: input_tensor})

            # Get logits and apply softmax
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            probs = exp_logits / np.sum(exp_logits)

            predicted_idx = np.argmax(probs)
            confidence = float(probs[predicted_idx])

            # ✅ SAFE VERSION (prevents crash/freeze)
            if predicted_idx >= len(self.EMOTION_LABELS):
                print(f"⚠️ Invalid prediction index: {predicted_idx}")
                return "sad", 0.0, probabilities  # safe fallback

            emotion = self.EMOTION_LABELS[predicted_idx]
            for i, label in enumerate(self.EMOTION_LABELS):
                if i < len(probs):
                    probabilities[label] = float(probs[i])

        except Exception as e:
            print(f"ONNX emotion detection error: {str(e)}")

        return emotion, confidence, probabilities

    def _detect_emotion_torchscript(self, processed_face):
        """Detect emotion using TorchScript model"""
        emotion = "neutral"
        confidence = 0.0
        probabilities = {label: 0.0 for label in self.EMOTION_LABELS}

        try:
            if self._torchscript_model is None:
                return emotion, confidence, probabilities

            # Convert to tensor: [1, 1, 48, 48]
            tensor = torch.FloatTensor(processed_face).unsqueeze(0).unsqueeze(0)

            # Run inference
            outputs = self._torchscript_model(tensor)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get results
            confidence, predicted_idx = torch.max(probs, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()

            if predicted_idx < len(self.EMOTION_LABELS):
                emotion = self.EMOTION_LABELS[predicted_idx]
                for i, label in enumerate(self.EMOTION_LABELS):
                    probabilities[label] = probs[0][i].item()

        except Exception as e:
            print(f"TorchScript emotion detection error: {str(e)}")

        return emotion, confidence, probabilities

    def _get_mock_emotion(self):
        """Fallback mock emotion detection"""
        import random
        emotions = ['happy', 'sad']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.7, 0.95)

        probabilities = {label: random.uniform(0, 0.3) for label in self.EMOTION_LABELS}
        probabilities[emotion] = confidence

    
        # Normalize probabilities
        total = sum(probabilities.values())
        for label in probabilities:
            probabilities[label] /= total

        return emotion, confidence, probabilities

    def _draw_on_image(self, image, rect, emotion, confidence):
        """Draw detection results on image"""
        try:
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)

            colors = {
                'angry': QColor(239, 68, 68),
                'disgust': QColor(16, 185, 129),
                'fear': QColor(139, 92, 246),
                'happy': QColor(245, 158, 11),
                'sad': QColor(59, 130, 246),
                'surprise': QColor(249, 115, 22),
                'neutral': QColor(148, 163, 184)
            }
            color = colors.get(emotion, QColor(255, 255, 255))

            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            label = f"{emotion}: {confidence:.1%}"
            font = QFont("Arial", 12, QFont.Bold)
            painter.setFont(font)

            text_rect = QRect(rect.x(), rect.y() - 30, rect.width(), 30)
            painter.fillRect(text_rect, QColor(0, 0, 0, 180))

            painter.setPen(Qt.white)
            painter.drawText(text_rect, Qt.AlignCenter, label)

            method_label = f"[{self._detection_method.value}]"
            method_font = QFont("Arial", 8)
            painter.setFont(method_font)
            method_rect = QRect(rect.x(), rect.y() + rect.height(), rect.width(), 20)
            painter.fillRect(method_rect, QColor(0, 0, 0, 150))
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(method_rect, Qt.AlignCenter, method_label)

            painter.end()

        except Exception as e:
            print(f"Drawing error: {str(e)}")

    def _record_detection(self, emotion, confidence, probabilities):
        """Record detection to session"""
        detection = {
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'method': self._detection_method.value
        }

        self._detections.append(detection)
        self._update_session_stats()

    def _update_session_stats(self):
        """Update session statistics"""
        total = len(self._detections)

        distribution = {}
        total_confidence = 0.0

        for detection in self._detections:
            emotion = detection['emotion']
            distribution[emotion] = distribution.get(emotion, 0) + 1
            total_confidence += detection['confidence']

        average_confidence = total_confidence / total if total > 0 else 0.0

        self._session_stats.update({
            'total_detections': total,
            'emotion_distribution': distribution,
            'average_confidence': average_confidence
        })

        self.session_stats_updated.emit(self._session_stats)

    def _update_fps(self):
        """Update FPS counter"""
        fps = self._frame_counter
        self._frame_counter = 0

        avg_process_time = np.mean(self._frame_times) if self._frame_times else 0
        actual_fps = 1.0 / avg_process_time if avg_process_time > 0 else fps

        self.fps_updated.emit(actual_fps)

    def _start_session(self):
        """Start recording session"""
        self._session_stats['start_time'] = datetime.now().isoformat()
        self._detections.clear()

    def _end_session(self):
        """End recording session"""
        self._session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats_updated.emit(self._session_stats)

    def _load_all_models(self):
        """Load all available emotion detection models"""
        self._available_methods = []
        models_loaded = 0

        # Try loading PyTorch model (state dict)
        if self._load_pytorch_model():
            self._available_methods.append("pytorch")
            models_loaded += 1
            self.model_status_changed.emit("PyTorch model loaded", True)

        # Try loading ONNX model
        if self._load_onnx_model():
            self._available_methods.append("onnx")
            models_loaded += 1
            self.model_status_changed.emit("ONNX model loaded", True)

        # Try loading TorchScript model
        if self._load_torchscript_model():
            self._available_methods.append("torchscript")
            models_loaded += 1
            self.model_status_changed.emit("TorchScript model loaded", True)

        # Set default method to first available\
        if self._available_methods:
        #   Prefer TorchScript since it's working correctly (2-class model)
            if "torchscript" in self._available_methods:
                self.set_detection_method("torchscript")
                print("Using TorchScript model (recommended)")
            else:
                self.set_detection_method(self._available_methods[0])
                print(f"Using fallback model: {self._available_methods[0]}")

            self._model_loaded = True
            self.model_loaded_changed.emit(True)
            self.model_status_changed.emit(f"Loaded {models_loaded} model(s)", True)
        else:
            self._model_loaded = False
            self.model_loaded_changed.emit(False)
            self.model_status_changed.emit("No models loaded - using mock data", False)

    def _create_emotion_cnn_model(self):
        """Create the EmotionCNN model architecture from your training script"""
        import torch.nn as nn
        import torch.nn.functional as F

        class EmotionCNN(nn.Module):
            def __init__(self, num_classes=2):
                super(EmotionCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(256)

                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.global_pool = nn.AdaptiveAvgPool2d((6, 6))

                self.fc1 = nn.Linear(256 * 6 * 6, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = self.pool(F.relu(self.bn4(self.conv4(x))))

                x = self.global_pool(x)
                x = x.view(-1, 256 * 6 * 6)

                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        return EmotionCNN(num_classes=2)

    def _load_pytorch_model(self):
        """Load PyTorch emotion detection model"""
        try:
            model_paths = [
                "exports/emotion_model_final.pth",
                "exports/emotion_model_complete.pth",
                "emotion_model_final.pth",
                "emotion_model_complete.pth"
            ]

            for model_path in model_paths:
                if Path(model_path).exists():
                    print(f"Loading PyTorch model from: {model_path}")

                    try:
                        # Create model architecture
                        model = self._create_emotion_cnn_model()

                        # Load state dict
                        if model_path.endswith('emotion_model_complete.pth'):
                            # Complete model with architecture
                            checkpoint = torch.load(model_path, map_location='cpu')
                            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                # Try to load as state dict directly
                                model.load_state_dict(checkpoint)
                        else:
                            # Regular state dict
                            state_dict = torch.load(model_path, map_location='cpu')
                            model.load_state_dict(state_dict)

                        model.eval()
                        self._pytorch_model = model

                        print(f"✓ PyTorch model loaded successfully from: {model_path}")

                        # Test with dummy input
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 1, 48, 48)
                            output = model(dummy_input)
                            print(f"  Test output shape: {output.shape}")

                        return True

                    except Exception as e:
                        print(f"Failed to load PyTorch model {model_path}: {str(e)}")
                        continue

            print("PyTorch model not found.")
            return False

        except Exception as e:
            print(f"Failed to load PyTorch model: {str(e)}")
            return False

    def _load_onnx_model(self):
        """Load ONNX emotion detection model"""
        try:
            model_paths = [
                "exports/emotion_model.onnx",
                "emotion_model.onnx"
            ]

            for model_path in model_paths:
                if Path(model_path).exists():
                    print(f"Loading ONNX model from: {model_path}")

                    try:
                        # Set providers for ONNX Runtime
                        providers = ['CPUExecutionProvider']

                        self._onnx_session = ort.InferenceSession(model_path, providers=providers)

                        # Verify model
                        input_name = self._onnx_session.get_inputs()[0].name
                        input_shape = self._onnx_session.get_inputs()[0].shape

                        print(f"✓ ONNX model loaded successfully")
                        print(f"  Input name: {input_name}")
                        print(f"  Input shape: {input_shape}")

                        return True

                    except Exception as e:
                        print(f"Failed to load ONNX model {model_path}: {str(e)}")
                        continue

            print("ONNX model not found.")
            return False

        except Exception as e:
            print(f"Failed to load ONNX model: {str(e)}")
            return False

    def _load_torchscript_model(self):
        """Load TorchScript model"""
        try:
            model_paths = [
                "exports/emotion_model_torchscript.pt",
                "emotion_model_torchscript.pt"
            ]

            for model_path in model_paths:
                if Path(model_path).exists():
                    print(f"Loading TorchScript model from: {model_path}")

                    try:
                        self._torchscript_model = torch.jit.load(model_path, map_location='cpu')
                        self._torchscript_model.eval()

                        print(f"✓ TorchScript model loaded successfully")

                        # Test with dummy input
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 1, 48, 48)
                            output = self._torchscript_model(dummy_input)
                            print(f"  Test output shape: {output.shape}")

                        return True

                    except Exception as e:
                        print(f"Failed to load TorchScript model {model_path}: {str(e)}")
                        continue

            print("TorchScript model not found.")
            return False

        except Exception as e:
            print(f"Failed to load TorchScript model: {str(e)}")
            return False

    def _init_face_cascade(self):
        """Initialize OpenCV face cascade"""
        try:
            cascade_paths = [
                "haarcascades/haarcascade_frontalface_default.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ]

            for path in cascade_paths:
                try:
                    if Path(path).exists():
                        self._face_cascade = cv2.CascadeClassifier(path)
                        if not self._face_cascade.empty():
                            print(f"Loaded face cascade from: {path}")
                            return
                except:
                    continue

            print("Warning: Could not load face cascade. Face detection will be disabled.")

        except Exception as e:
            print(f"Failed to load face cascade: {str(e)}")

    def _init_face_dnn(self):
        """Initialize OpenCV DNN face detector"""
        try:
            proto_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

            if Path(proto_path).exists() and Path(model_path).exists():
                self._face_dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self._face_detection_method = "dnn"
                print("Loaded DNN face detector")

        except Exception as e:
            print(f"Failed to load DNN face detector: {str(e)}")

    def _update_available_cameras(self):
        """Update list of available cameras"""
        try:
            devices = QMediaDevices.videoInputs()
            cameras = [device.description() for device in devices]
            self.available_cameras_changed.emit(cameras)
        except Exception as e:
            print(f"Failed to get cameras: {str(e)}")

class FrameProcessor(QThread):
    frame_ready = Signal(QImage)
    emotion_detected = Signal(str, float, dict)
    face_count_updated = Signal(int)

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.running = True
        self.frame = None

    def run(self):
        while self.running:
            if self.frame is not None:
                # ✅ _process_frame expects a QImage
                self.backend._process_frame(self.frame)
                self.frame = None
            self.msleep(5)

    @Slot(QVideoFrame)
    def submit_frame(self, video_frame):
        if video_frame is None or not video_frame.isValid():
            return

        # Convert to QImage only once
        image = video_frame.toImage()
        if image is None or image.isNull():
            return

        self.frame = image

    def stop(self):
        self.running = False
        self.wait()