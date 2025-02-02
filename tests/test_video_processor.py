import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.video_processor = VideoProcessor()
    
    def tearDown(self):
        """Clean up after each test method."""
        if self.video_processor.is_running:
            self.video_processor.stop_capture()
    
    def test_initialization(self):
        """Test if VideoProcessor initializes with correct default values."""
        self.assertIsNone(self.video_processor.capture)
        self.assertIsNone(self.video_processor.processing_thread)
        self.assertFalse(self.video_processor.is_running)
        self.assertIsNone(self.video_processor.current_frame)
        self.assertIsNone(self.video_processor.object_detector)
        self.assertIsNone(self.video_processor.face_analyzer)
    
    @patch('cv2.VideoCapture')
    def test_start_capture_success(self, mock_capture):
        """Test successful video capture initialization."""
        # Mock successful camera opening
        mock_capture.return_value.isOpened.return_value = True
        
        # Test start_capture
        result = self.video_processor.start_capture(0)
        
        self.assertTrue(result)
        self.assertTrue(self.video_processor.is_running)
        self.assertIsNotNone(self.video_processor.processing_thread)
        mock_capture.assert_called_once_with(0)
    
    @patch('cv2.VideoCapture')
    def test_start_capture_failure(self, mock_capture):
        """Test video capture initialization failure."""
        # Mock failed camera opening
        mock_capture.return_value.isOpened.return_value = False
        
        # Test start_capture
        result = self.video_processor.start_capture(0)
        
        self.assertFalse(result)
        self.assertFalse(self.video_processor.is_running)
        self.assertIsNone(self.video_processor.processing_thread)
    
    def test_stop_capture(self):
        """Test stopping video capture."""
        # Mock the capture and thread
        self.video_processor.capture = MagicMock()
        self.video_processor.processing_thread = MagicMock()
        self.video_processor.is_running = True
        
        # Stop capture
        self.video_processor.stop_capture()
        
        self.assertFalse(self.video_processor.is_running)
        self.video_processor.capture.release.assert_called_once()
        self.video_processor.processing_thread.join.assert_called_once()

if __name__ == '__main__':
    unittest.main()