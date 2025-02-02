import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from collections import deque

class FaceAnalyzer:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """Initialize the face analysis module.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with more detailed settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,  # Enable detailed landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Enhanced temporal analysis with multi-scale tracking
        self.temporal_window = 30  # Store last 30 frames for smooth tracking
        self.landmark_history = deque(maxlen=self.temporal_window)
        self.emotion_history = deque(maxlen=self.temporal_window)
        self.movement_history = deque(maxlen=self.temporal_window)  # Track facial movements
        
        # Movement tracking parameters
        self.movement_threshold = 0.001  # Threshold for micro-movements
        self.smoothing_factor = 0.3  # Temporal smoothing factor
        self.frame_count = 0
        self.skip_frames = 1  # Process every frame by default
        
        # Initialize emotion recognition model
        self.emotion_model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self._initialize_emotion_model()
    
    def _initialize_emotion_model(self):
        try:
            # Define emotion recognition model
            class EmotionNet(nn.Module):
                def __init__(self, num_emotions=7):
                    super().__init__()
                    self.base_model = resnet18(pretrained=True)
                    # Add temporal features
                    self.lstm = nn.LSTM(512, 256, num_layers=2, batch_first=True)
                    self.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(256, num_emotions),
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    batch_size = x.size(0)
                    # Extract features
                    features = self.base_model.conv1(x)
                    features = self.base_model.bn1(features)
                    features = self.base_model.relu(features)
                    features = self.base_model.maxpool(features)
                    features = self.base_model.layer1(features)
                    features = self.base_model.layer2(features)
                    features = self.base_model.layer3(features)
                    features = self.base_model.layer4(features)
                    features = self.base_model.avgpool(features)
                    features = features.view(batch_size, -1)
                    
                    # Process temporal sequence
                    features = features.unsqueeze(1)  # Add time dimension
                    lstm_out, _ = self.lstm(features)
                    return self.fc(lstm_out[:, -1, :])
            
            self.emotion_model = EmotionNet()
            self.emotion_model.eval()
            if torch.cuda.is_available():
                self.emotion_model.cuda()
            
            # Define image transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error initializing emotion model: {str(e)}")
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Analyze faces in the input frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Tuple containing:
            - Annotated frame with facial landmarks and emotions
            - List of face analysis results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Convert back to BGR for drawing
        annotated_frame = frame.copy()
        
        face_results = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract and store temporal data
                face_data = self._extract_face_data(face_landmarks, frame.shape)
                self.landmark_history.append(face_data['landmarks'])
                
                # Analyze facial features and emotions
                emotions = self._analyze_emotions(frame, face_data)
                self.emotion_history.append(emotions)
                
                # Detect microexpressions
                micro_expressions = self._detect_microexpressions()
                
                # Draw detailed facial mesh
                self._draw_detailed_mesh(annotated_frame, face_landmarks)
                
                face_results.append({
                    'landmarks': face_data['landmarks'],
                    'bbox': face_data['bbox'],
                    'emotions': emotions,
                    'micro_expressions': micro_expressions
                })
                
                # Draw emotion and microexpression labels
                self._draw_analysis_results(annotated_frame, face_data['bbox'], 
                                          emotions, micro_expressions)
        
        return annotated_frame, face_results
    
    def _draw_detailed_mesh(self, frame: np.ndarray, landmarks) -> None:
        # Draw full face mesh
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), 
                                                           thickness=1, 
                                                           circle_radius=1),
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style()
        )
        
        # Draw contours for specific facial features
        feature_specs = [
            (self.mp_face_mesh.FACEMESH_LEFT_EYE, (0, 255, 0)),  # Green for eyes
            (self.mp_face_mesh.FACEMESH_RIGHT_EYE, (0, 255, 0)),
            (self.mp_face_mesh.FACEMESH_LIPS, (0, 0, 255)),      # Red for lips
            (self.mp_face_mesh.FACEMESH_LEFT_EYEBROW, (255, 0, 0)),  # Blue for eyebrows
            (self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW, (255, 0, 0))
        ]
        
        for connections, color in feature_specs:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=color, thickness=2)
            )
    
    def _detect_microexpressions(self) -> Dict[str, float]:
        if len(self.landmark_history) < 2:
            return {}
        
        # Analyze temporal changes in key facial features with smoothing
        current = np.array(self.landmark_history[-1])
        previous = np.array(self.landmark_history[-2])
        
        # Calculate movement deltas with temporal smoothing
        deltas = np.abs(current - previous) * self.smoothing_factor
        
        # Define regions for micro-movement detection with specific landmark indices
        regions = {
            'brow': slice(0, 20),    # Eyebrow region
            'eye': slice(20, 40),     # Eye region
            'nose': slice(40, 60),    # Nose region
            'mouth': slice(60, 80),   # Mouth region
            'cheek': slice(80, 100)   # Cheek region
        }
        
        # Detect subtle movements in each region with adaptive thresholding
        micro_movements = {}
        for region, indices in regions.items():
            # Calculate regional movement with weighted average
            movement = np.mean(deltas[indices])
            
            # Apply adaptive thresholding based on movement history
            if len(self.movement_history) > 0:
                historical_avg = np.mean([m.get(region, 0) for m in self.movement_history])
                threshold = max(self.movement_threshold, historical_avg * 0.5)
            else:
                threshold = self.movement_threshold
            
            # Detect micro-movements with dynamic thresholding
            if movement > threshold and movement < threshold * 10:
                micro_movements[region] = float(movement)
                
            # Track rapid changes
            if movement > threshold * 10:
                micro_movements[f"{region}_rapid"] = float(movement)
        
        # Update movement history for temporal analysis
        self.movement_history.append(micro_movements)
        
        return micro_movements
    
    def _draw_analysis_results(self, frame, bbox, emotions, micro_expressions):
        x1, y1, x2, y2 = bbox
        
        # Draw emotion results with confidence bar
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_text = f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}"
        cv2.putText(frame, emotion_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw micro-expression indicators with visual feedback
        if micro_expressions:
            y_offset = y1 - 30
            for region, intensity in micro_expressions.items():
                # Color coding based on movement intensity
                if "_rapid" in region:
                    color = (0, 0, 255)  # Red for rapid movements
                    region = region.replace("_rapid", " (Rapid)")
                else:
                    # Gradient from yellow to red based on intensity
                    intensity_factor = min(intensity / 0.01, 1.0)
                    color = (0, int(255 * (1 - intensity_factor)), int(255 * intensity_factor))
                
                micro_text = f"{region.capitalize()}: {intensity:.3f}"
                cv2.putText(frame, micro_text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw intensity bar
                bar_length = int(100 * intensity / 0.01)
                bar_length = min(100, bar_length)
                cv2.rectangle(frame, 
                            (x1 + 150, y_offset - 10),
                            (x1 + 150 + bar_length, y_offset - 5),
                            color, -1)
                
                y_offset -= 20
    
    def _extract_face_data(self, landmarks, frame_shape) -> Dict:
        """Extract face region and landmark coordinates."""
        h, w = frame_shape[:2]
        landmarks_2d = []
        x_coords = []
        y_coords = []
        
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks_2d.append([x, y])
            x_coords.append(x)
            y_coords.append(y)
        
        # Calculate bounding box
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        return {
            'landmarks': landmarks_2d,
            'bbox': [x1, y1, x2, y2]
        }
    
    def _analyze_emotions(self, frame: np.ndarray, face_data: Dict) -> Dict[str, float]:
        """Analyze facial expressions and emotions using deep learning model.
        Detects both macro and micro expressions."""
        if self.emotion_model is None:
            # Initialize emotion recognition model
            try:
                import torch
                import torch.nn as nn
                import torchvision.transforms as transforms
                from torchvision.models import resnet18
                
                # Define emotion recognition model
                class EmotionNet(nn.Module):
                    def __init__(self, num_emotions=7):
                        super().__init__()
                        self.base_model = resnet18(pretrained=True)
                        self.base_model.fc = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(512, num_emotions),
                            nn.Softmax(dim=1)
                        )
                    
                    def forward(self, x):
                        return self.base_model(x)
                
                # Load pre-trained model (you would need to provide the actual model weights)
                self.emotion_model = EmotionNet()
                self.emotion_model.eval()
                if torch.cuda.is_available():
                    self.emotion_model.cuda()
                
                # Define image transforms
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
                
                self.emotion_labels = ['angry', 'disgust', 'fear', 'happy',
                                      'sad', 'surprise', 'neutral']
            except Exception as e:
                print(f"Error initializing emotion model: {str(e)}")
                return {'neutral': 1.0}
        
        try:
            # Extract face region
            x1, y1, x2, y2 = face_data['bbox']
            face_img = frame[y1:y2, x1:x2]
            
            # Preprocess image
            face_tensor = self.transform(face_img)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            face_tensor = face_tensor.unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.emotion_model(face_tensor)
                probs = outputs[0].cpu().numpy()
            
            # Create emotion dictionary
            emotions = {}
            for label, prob in zip(self.emotion_labels, probs):
                emotions[label] = float(prob)
            
            # Detect microexpressions by analyzing temporal changes
            # TODO: Implement microexpression detection using temporal analysis
            
            return emotions
        except Exception as e:
            print(f"Error analyzing emotions: {str(e)}")
            return {'neutral': 1.0}
    
    def _draw_emotion_labels(self, frame: np.ndarray, bbox: List[int], emotions: Dict[str, float]) -> None:
        """Draw emotion labels on the frame."""
        x1, y1, x2, y2 = bbox
        
        # Get the dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        label = f"{dominant_emotion[0]}: {dominant_emotion[1]:.2f}"
        
        # Draw the label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)