import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict

class ObjectDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25):
        """Initialize the YOLO-based object detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Perform object detection on the input frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Tuple containing:
            - Annotated frame with detection boxes
            - List of detection results with class, confidence, and coordinates
        """
        if self.model is None:
            return frame, []
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls = int(box.cls)
                conf = float(box.conf)
                
                detections.append({
                    'class': self.model.names[cls],
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        return annotated_frame, detections
    
    def optimize_model(self):
        """Optimize the model for inference using TensorRT if available."""
        if self.model and self.device == 'cuda':
            try:
                # Export to TensorRT format
                self.model.export(format='engine')
            except Exception as e:
                print(f"Error optimizing model: {str(e)}")