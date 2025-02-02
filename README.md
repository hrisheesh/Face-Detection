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

3. Download the YOLOv3 weights:
```bash
python download_weights.py
```

Note: The YOLOv3 weights file is large (236.52 MB) and is not included in the repository. The script will download it automatically.

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