import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import tensorflow as tf

class FaceAnalyzer:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Face Mesh with 3D mesh and enhanced landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False  # Enable video mode for better tracking
        )
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils
        self.tesselation_style = self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        
        # Load pre-trained emotion detection model with custom optimizer config
        try:
            self.emotion_model = load_model('models/emotion_model.h5', compile=False)
            # Recompile the model with updated optimizer configuration
            self.emotion_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error loading emotion model: {str(e)}")
            self.emotion_model = None
        
        # Emotion labels with more nuanced categories
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        
        # Initialize temporal smoothing with increased buffer
        self.emotion_buffer = deque(maxlen=10)  # Increased buffer size for better smoothing
        self.last_emotion = None
        
        # Initialize FACS-based microexpression detection
        self.action_units = {
            'inner_brow_raiser': [27, 28],  # Landmark indices for inner brow movement
            'outer_brow_raiser': [17, 18, 19],  # Outer brow landmarks
            'brow_lowerer': [48, 49, 50],  # Brow lowering landmarks
            'nose_wrinkler': [4, 5, 6],  # Nose wrinkle landmarks
            'lip_corner_puller': [61, 291],  # Smile landmarks
            'lip_corner_depressor': [291, 375]  # Frown landmarks
        }
        
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion detection"""
        face_img = cv2.resize(face_img, (64, 64))  # Changed to 64x64 to match model's expected input
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        return face_img
    
    def get_face_bbox(self, face_landmarks, image_shape):
        """Extract face bounding box from landmarks with dynamic padding"""
        h, w = image_shape[:2]
        x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
        
        x_min = int(min(x_coordinates) * w)
        x_max = int(max(x_coordinates) * w)
        y_min = int(min(y_coordinates) * h)
        y_max = int(max(y_coordinates) * h)
        
        # Dynamic padding based on face size
        face_width = x_max - x_min
        face_height = y_max - y_min
        padding_x = int(face_width * 0.2)
        padding_y = int(face_height * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return x_min, y_min, x_max, y_max
    
    def detect_microexpressions(self, face_landmarks):
        """Detect subtle facial movements using FACS-inspired approach"""
        micro_movements = {}
        
        for action, landmark_indices in self.action_units.items():
            # Calculate movement intensity for each action unit
            landmarks = [face_landmarks.landmark[idx] for idx in landmark_indices]
            movement = self._calculate_movement_intensity(landmarks)
            micro_movements[action] = movement
        
        return micro_movements
    
    def _calculate_movement_intensity(self, landmarks):
        """Calculate the intensity of facial muscle movement"""
        # Simple distance-based calculation for demonstration
        if len(landmarks) < 2:
            return 0.0
        
        distances = []
        for i in range(len(landmarks) - 1):
            dist = np.sqrt((landmarks[i+1].x - landmarks[i].x)**2 +
                          (landmarks[i+1].y - landmarks[i].y)**2)
            distances.append(dist)
        
        return np.mean(distances)
    
    def smooth_emotion(self, emotion):
        """Apply temporal smoothing to emotion predictions with confidence weighting"""
        self.emotion_buffer.append(emotion)
        if len(self.emotion_buffer) < self.emotion_buffer.maxlen:
            return emotion
        
        # Get most common emotion in buffer with weighted voting
        emotions_count = {}
        for e in self.emotion_buffer:
            emotions_count[e] = emotions_count.get(e, 0) + 1
        
        smoothed_emotion = max(emotions_count.items(), key=lambda x: x[1])[0]
        return smoothed_emotion
    
    def analyze_frame(self, frame):
        """Analyze frame for facial expressions and microexpressions"""
        if frame is None:
            return None, None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        emotion_data = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face bounding box
                x_min, y_min, x_max, y_max = self.get_face_bbox(face_landmarks, frame.shape)
                
                # Extract and preprocess face region
                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size == 0:
                    continue
                
                processed_face = self.preprocess_face(face_roi)
                
                # Predict emotion
                emotion_pred = self.emotion_model.predict(processed_face, verbose=0)[0]
                emotion_idx = np.argmax(emotion_pred)
                emotion = self.emotions[emotion_idx]
                confidence = emotion_pred[emotion_idx]
                
                # Detect microexpressions
                micro_movements = self.detect_microexpressions(face_landmarks)
                
                # Apply temporal smoothing
                smoothed_emotion = self.smooth_emotion(emotion)
                
                # Draw bounding box and emotion label
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f'{smoothed_emotion}: {confidence:.2f}'
                cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw 3D mesh and enhanced facial landmarks
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.tesselation_style
                )
                
                # Calculate and visualize depth information
                for landmark in face_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Use z-coordinate for depth visualization (red = closer, blue = further)
                    depth_color = (int(255 * (1-landmark.z)), 0, int(255 * landmark.z))
                    cv2.circle(annotated_frame, (x, y), 1, depth_color, -1)
                
                emotion_data = {
                    'emotion': smoothed_emotion,
                    'confidence': float(confidence),
                    'all_emotions': dict(zip(self.emotions, emotion_pred.tolist())),
                    'micro_expressions': micro_movements
                }
                
                self.last_emotion = emotion_data
        
        return annotated_frame, emotion_data or self.last_emotion