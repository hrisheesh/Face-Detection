import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class FaceAnalyzer:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Face Mesh with optimized settings for better tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )
        
        # Optimize landmark smoothing parameters
        self.landmark_buffer = deque(maxlen=3)  # Reduced buffer size for faster response
        self.smoothing_factor = 0.7  # Increased smoothing factor for stability
        
        # Enhanced face tracking parameters
        self.face_trackers = {}
        self.next_face_id = 0
        self.face_timeout = 10  # Reduced for faster cleanup
        self.min_face_size = 50
        self.iou_threshold = 0.5
        
        # Initialize landmark smoothing
        self.landmark_buffer = deque(maxlen=5)  # Buffer for temporal smoothing
        self.last_landmarks = None
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced drawing styles for better visualization
        self.mesh_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=1
        )
        
        # Camera matrix for 3D calculations
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1))
    
    def _enhance_mesh_visibility(self, frame, landmarks):
        """Enhance mesh visibility with optimized drawing parameters"""
        # Draw enhanced face mesh with better visibility and smoother lines
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=1,
                circle_radius=1
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=1
            )
        )
        
        # Draw contours with improved visibility
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    def _calculate_face_orientation(self, landmarks):
        """Calculate face orientation using facial landmarks with improved angle detection"""
        # Get key facial landmarks for orientation calculation
        nose_tip = landmarks.landmark[4]  # Nose tip
        left_eye = landmarks.landmark[33]  # Left eye
        right_eye = landmarks.landmark[263]  # Right eye
        mouth_left = landmarks.landmark[61]  # Left mouth corner
        mouth_right = landmarks.landmark[291]  # Right mouth corner
        
        # Calculate face center with weighted contribution
        face_points = [left_eye, right_eye, nose_tip, mouth_left, mouth_right]
        weights = [1.0, 1.0, 1.5, 0.75, 0.75]  # Give more weight to nose tip for better angle detection
        face_center = np.average([[p.x, p.y, p.z] for p in face_points], axis=0, weights=weights)
        
        # Calculate face normal vector using enhanced cross product method
        eye_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y, right_eye.z - left_eye.z])
        nose_vector = np.array([nose_tip.x - face_center[0], nose_tip.y - face_center[1], nose_tip.z - face_center[2]])
        
        # Normalize vectors for more stable calculations
        eye_vector = eye_vector / (np.linalg.norm(eye_vector) + 1e-6)
        nose_vector = nose_vector / (np.linalg.norm(nose_vector) + 1e-6)
        
        face_normal = np.cross(eye_vector, nose_vector)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-6)
        
        # Calculate Euler angles with improved stability
        pitch = np.arctan2(face_normal[1], np.sqrt(face_normal[0]**2 + face_normal[2]**2))
        yaw = np.arctan2(face_normal[0], face_normal[2])
        roll = np.arctan2(eye_vector[1], eye_vector[0])
        
        # Convert to degrees
        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))
        
        # Check if face is turned too much (>35 degrees)
        is_extreme_angle = abs(yaw_deg) > 35
        
        return {
            'pitch': pitch_deg,
            'yaw': yaw_deg,
            'roll': roll_deg,
            'is_extreme_angle': is_extreme_angle
        }

    def smooth_landmarks(self, landmarks):
        """Apply enhanced temporal smoothing to landmarks with velocity prediction"""
        if not landmarks:
            return self.last_landmarks
        
        self.landmark_buffer.append(landmarks)
        if len(self.landmark_buffer) < 3:
            return landmarks
        
        # Enhanced Kalman-like smoothing with velocity prediction
        weights = np.exp(np.linspace(-2, 0, len(self.landmark_buffer)))
        weights = weights / np.sum(weights)
        
        # Calculate velocities for prediction
        velocities = []
        for i in range(len(self.landmark_buffer) - 1):
            curr = self.landmark_buffer[i]
            next_frame = self.landmark_buffer[i + 1]
            vel = []
            for j in range(len(curr.landmark)):
                vx = next_frame.landmark[j].x - curr.landmark[j].x
                vy = next_frame.landmark[j].y - curr.landmark[j].y
                vz = next_frame.landmark[j].z - curr.landmark[j].z
                vel.append((vx, vy, vz))
            velocities.append(vel)
        
        # Predict next position using velocity
        if velocities:
            avg_velocity = np.average(np.array(velocities), axis=0, weights=weights[:-1])
            for i in range(len(landmarks.landmark)):
                # Weighted average of positions
                positions = np.array([[buffer.landmark[i].x, buffer.landmark[i].y, buffer.landmark[i].z] 
                                    for buffer in self.landmark_buffer])
                smoothed_position = np.average(positions, axis=0, weights=weights)
                
                # Add predicted velocity component
                smoothed_position += avg_velocity[i] * 0.5  # Damping factor
                
                landmarks.landmark[i].x = float(smoothed_position[0])
                landmarks.landmark[i].y = float(smoothed_position[1])
                landmarks.landmark[i].z = float(smoothed_position[2])
        
        self.last_landmarks = landmarks
        return landmarks
    
    def analyze_frame(self, frame):
        """Analyze frame for facial landmarks and mesh"""
        if frame is None:
            return None, None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        face_data = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Apply temporal smoothing to landmarks
                smoothed_landmarks = self.smooth_landmarks(face_landmarks)
                if smoothed_landmarks is None:
                    continue
                
                # Apply enhanced mesh visualization
                self._enhance_mesh_visibility(annotated_frame, smoothed_landmarks)
                
                # Calculate face orientation
                orientation = self._calculate_face_orientation(smoothed_landmarks)
                
                face_data = {
                    'landmarks': [[l.x, l.y, l.z] for l in smoothed_landmarks.landmark],
                    'orientation': orientation
                }
        
        return annotated_frame, face_data
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0