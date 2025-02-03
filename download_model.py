import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    models_dir = 'models'
    model_file = os.path.join(models_dir, 'emotion_model.h5')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Pre-trained emotion detection model URL
    url = 'https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    
    print(f'Downloading emotion detection model to {model_file}...')
    download_file(url, model_file)
    print('Download complete!')

if __name__ == '__main__':
    main()