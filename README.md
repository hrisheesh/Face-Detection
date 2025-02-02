# Face-Detection

A comprehensive facial analysis system with real-time emotion detection and micro-expression analysis capabilities.

## Features

- Real-time face detection and tracking
- Facial landmark detection with detailed mesh visualization
- Emotion recognition (7 basic emotions)
- Micro-expression detection and analysis
- Temporal analysis for smooth tracking
- GPU acceleration support (when available)

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- PyTorch
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hrisheesh/Face-Detection.git
cd Face-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `src/core/`: Core functionality modules
  - `face_analyzer.py`: Main face analysis implementation
  - `video_processor.py`: Video capture and processing
  - `object_detector.py`: Object detection utilities
- `src/ui/`: User interface components
- `tests/`: Unit tests
- `models/`: Pre-trained model weights and configurations

## License

MIT License

## Author

hrisheesh