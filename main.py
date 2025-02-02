import sys
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.core.video_processor import VideoProcessor

def main():
    # Initialize the application
    app = QApplication(sys.argv)
    
    # Create video processor instance
    video_processor = VideoProcessor()
    
    # Create and show the main window
    main_window = MainWindow(video_processor)
    main_window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()