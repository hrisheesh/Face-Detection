import cv2
import numpy as np
from .face_analyzer import FaceAnalyzer
from typing import Optional, Tuple

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.face_analyzer = FaceAnalyzer(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.process_interval = 1.0 / 30  # Default 30 FPS processing
        self.last_process_time = 0
    
    def start_capture(self, camera_id: int = 0) -> bool:
        """Start video capture from specified camera."""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def stop_capture(self) -> None:
        """Stop video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame with facial analysis."""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Process frame with facial analysis
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if (current_time - self.last_process_time) >= self.process_interval:
            # Perform facial analysis
            processed_frame, _ = self.face_analyzer.analyze_frame(frame)
            self.last_process_time = current_time
            return processed_frame
        
        return frame
    
    def set_detection_confidence(self, confidence: float) -> None:
        """Update face detection confidence threshold."""
        if hasattr(self.face_analyzer, 'face_mesh'):
            self.face_analyzer = FaceAnalyzer(
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
            )
    
    def set_processing_interval(self, fps: float) -> None:
        """Set the processing interval based on desired FPS."""
        if fps > 0:
            self.process_interval = 1.0 / fps