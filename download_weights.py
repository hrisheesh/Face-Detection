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
    weights_dir = 'models'
    weights_file = os.path.join(weights_dir, 'yolov3.weights')
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # YOLOv3 weights URL
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    
    print(f'Downloading YOLOv3 weights to {weights_file}...')
    download_file(url, weights_file)
    print('Download complete!')

if __name__ == '__main__':
    main()