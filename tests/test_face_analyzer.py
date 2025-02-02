import unittest
import numpy as np
import mediapipe as mp
from unittest.mock import MagicMock, patch
from src.core.face_analyzer import FaceAnalyzer

class TestFaceAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.face_analyzer = FaceAnalyzer()
    
    def test_initialization(self):
        """Test if FaceAnalyzer initializes with correct default values."""
        self.assertIsNotNone(self.face_analyzer.mp_face_mesh)
        self.assertIsNotNone(self.face_analyzer.mp_drawing)
        self.assertIsNotNone(self.face_analyzer.face_mesh)
        self.assertIsNone(self.face_analyzer.emotion_model)
    
    @patch('mediapipe.solutions.face_mesh.FaceMesh')
    def test_face_mesh_initialization(self, mock_face_mesh):
        """Test face mesh initialization with custom parameters."""
        custom_detection_conf = 0.7
        custom_tracking_conf = 0.8
        
        analyzer = FaceAnalyzer(
            min_detection_confidence=custom_detection_conf,
            min_tracking_confidence=custom_tracking_conf
        )
        
        mock_face_mesh.assert_called_once_with(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=custom_detection_conf,
            min_tracking_confidence=custom_tracking_conf
        )
    
    def test_extract_face_data(self):
        """Test face data extraction from landmarks."""
        # Create mock landmarks
        mock_landmarks = MagicMock()
        mock_landmarks.landmark = [
            MagicMock(x=0.4, y=0.4),
            MagicMock(x=0.6, y=0.6)
        ]
        
        frame_shape = (100, 100, 3)
        face_data = self.face_analyzer._extract_face_data(mock_landmarks, frame_shape)
        
        self.assertIn('landmarks', face_data)
        self.assertIn('bbox', face_data)
        self.assertEqual(len(face_data['landmarks']), 2)
        self.assertEqual(len(face_data['bbox']), 4)
    
    @patch('torch.cuda.is_available')
    def test_emotion_model_initialization(self, mock_cuda_available):
        """Test emotion model initialization."""
        mock_cuda_available.return_value = False
        
        # Test initial state
        self.assertIsNone(self.face_analyzer.emotion_model)
        
        # Trigger model initialization
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emotions = self.face_analyzer._analyze_emotions(
            test_frame,
            {'bbox': [0, 0, 100, 100]}
        )
        
        # Verify model was initialized
        self.assertIsNotNone(self.face_analyzer.emotion_model)
        self.assertIsInstance(emotions, dict)
        self.assertGreater(len(emotions), 0)
    
    def test_analyze_frame_no_faces(self):
        """Test frame analysis when no faces are detected."""
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face mesh process to return no faces
        self.face_analyzer.face_mesh.process = MagicMock(
            return_value=MagicMock(multi_face_landmarks=None)
        )
        
        frame_out, face_results = self.face_analyzer.analyze_frame(test_frame)
        
        self.assertTrue(np.array_equal(frame_out, test_frame))
        self.assertEqual(len(face_results), 0)

if __name__ == '__main__':
    unittest.main()