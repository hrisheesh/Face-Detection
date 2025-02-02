import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.object_detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.object_detector = ObjectDetector()
    
    def test_initialization(self):
        """Test if ObjectDetector initializes with correct default values."""
        self.assertEqual(self.object_detector.conf_threshold, 0.25)
        self.assertIn(self.object_detector.device, ['cuda', 'cpu'])
    
    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda_available):
        """Test device selection based on CUDA availability."""
        # Test when CUDA is available
        mock_cuda_available.return_value = True
        detector = ObjectDetector()
        self.assertEqual(detector.device, 'cuda')
        
        # Test when CUDA is not available
        mock_cuda_available.return_value = False
        detector = ObjectDetector()
        self.assertEqual(detector.device, 'cpu')
    
    @patch('ultralytics.YOLO')
    def test_model_loading(self, mock_yolo):
        """Test YOLO model loading."""
        # Test successful model loading
        detector = ObjectDetector()
        mock_yolo.assert_called_once_with('yolov8n.pt')
        
        # Test model loading with custom weights
        custom_weights = 'custom_model.pt'
        detector = ObjectDetector(model_path=custom_weights)
        mock_yolo.assert_called_with(custom_weights)
    
    def test_detect_no_model(self):
        """Test detection when model is not loaded."""
        self.object_detector.model = None
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_out, detections = self.object_detector.detect(test_frame)
        
        self.assertTrue(np.array_equal(test_frame, frame_out))
        self.assertEqual(len(detections), 0)
    
    @patch('ultralytics.YOLO')
    def test_detect_with_objects(self, mock_yolo):
        """Test object detection with mock results."""
        # Create mock detection results
        mock_box = MagicMock()
        mock_box.xyxy = [np.array([100, 100, 200, 200])]
        mock_box.cls = np.array([0])
        mock_box.conf = np.array([0.95])
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        
        self.object_detector.model = mock_model
        
        # Test detection
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_out, detections = self.object_detector.detect(test_frame)
        
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['class'], 'person')
        self.assertEqual(detections[0]['confidence'], 0.95)
        self.assertEqual(detections[0]['bbox'], [100, 100, 200, 200])

if __name__ == '__main__':
    unittest.main()