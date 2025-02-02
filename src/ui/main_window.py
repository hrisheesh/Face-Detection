from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2

class MainWindow(QMainWindow):
    def __init__(self, video_processor):
        super().__init__()
        self.video_processor = video_processor
        self.init_ui()
        # Start camera automatically
        self.toggle_camera()
        
    def init_ui(self):
        # Set window properties with modern styling
        self.setWindowTitle('Advanced Face Detection')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: #ffffff; background-color: #1e1e1e; border-radius: 5px; padding: 5px; }
            QPushButton { 
                background-color: #0d47a1; color: white; border-radius: 4px; 
                padding: 8px 15px; font-weight: bold; min-width: 100px;
            }
            QPushButton:hover { background-color: #1565c0; }
            QPushButton:pressed { background-color: #0a3d91; }
            QSlider::handle:horizontal { background: #0d47a1; border-radius: 7px; }
            QSlider::groove:horizontal { background: #424242; height: 4px; }
            QComboBox { 
                background-color: #424242; color: white; border-radius: 4px; 
                padding: 5px; min-width: 100px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create video display label with improved styling
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #424242;")
        main_layout.addWidget(self.video_label)
        
        # Create controls container
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        self.start_button = QPushButton('Start Camera')
        self.start_button.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.start_button)
        
        # Add FPS control
        fps_layout = QHBoxLayout()
        fps_label = QLabel('FPS:')
        fps_label.setStyleSheet("background: transparent; min-width: 40px;")
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setStyleSheet("""
            QSpinBox { 
                background-color: #424242; color: white; 
                border-radius: 4px; padding: 5px;
            }
        """)
        self.fps_spinbox.valueChanged.connect(self.update_fps)
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_spinbox)
        camera_controls.addLayout(fps_layout)
        
        # Add detection confidence threshold control
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel('Confidence:')
        confidence_label.setStyleSheet("background: transparent; min-width: 80px;")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(70)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        camera_controls.addLayout(confidence_layout)
        
        # Add detection mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel('Mode:')
        mode_label.setStyleSheet("background: transparent; min-width: 50px;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Normal', 'Performance', 'Accuracy'])
        self.mode_combo.currentTextChanged.connect(self.update_detection_mode)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        camera_controls.addLayout(mode_layout)
        
        # Add all controls to main layout
        controls_layout.addLayout(camera_controls)
        main_layout.addLayout(controls_layout)
        
        # Initialize camera state
        self.camera_active = False
        
        # Set up timer for video updates with reduced frequency
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(50)  # 20 FPS default for smoother performance
        
    def update_fps(self, value):
        if value > 0:
            self.timer.setInterval(int(1000 / value))
            
    def update_confidence(self, value):
        threshold = value / 100.0
        if hasattr(self.video_processor, 'face_detection'):
            self.video_processor.face_detection = self.video_processor.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=threshold
            )
            
    def update_detection_mode(self, mode):
        if hasattr(self.video_processor, 'face_detection'):
            if mode == 'Performance':
                self.video_processor.process_interval = 1.0 / 15  # 15 FPS
                self.fps_spinbox.setValue(15)
            elif mode == 'Accuracy':
                self.video_processor.process_interval = 1.0 / 24  # 24 FPS
                self.fps_spinbox.setValue(24)
            else:  # Normal
                self.video_processor.process_interval = 1.0 / 30  # 30 FPS
                self.fps_spinbox.setValue(30)

    def toggle_camera(self):
        if not self.camera_active:
            # Start camera
            if self.video_processor.start_capture():
                self.camera_active = True
                self.start_button.setText('Stop Camera')
                self.timer.start()
        else:
            # Stop camera
            self.video_processor.stop_capture()
            self.camera_active = False
            self.start_button.setText('Start Camera')
            self.timer.stop()
            self.video_label.clear()
    
    def update_frame(self):
        frame = self.video_processor.get_current_frame()
        if frame is not None:
            try:
                # Convert frame to QImage for display
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                # Convert BGR to RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Scale image to fit label while maintaining aspect ratio
                # Use FastTransformation for better performance
                label_size = self.video_label.size()
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    label_size,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation
                )
            except Exception as e:
                print(f"Error updating frame: {e}")
                return
            
            # Center the image in the label
            x = (label_size.width() - scaled_pixmap.width()) // 2
            y = (label_size.height() - scaled_pixmap.height()) // 2
            self.video_label.setContentsMargins(x, y, x, y)
            self.video_label.setPixmap(scaled_pixmap)
    
    def closeEvent(self, event):
        # Clean up resources when window is closed
        self.video_processor.stop_capture()
        event.accept()