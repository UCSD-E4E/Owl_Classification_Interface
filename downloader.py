import sys
import requests
import gdown

FrozenModelURL = "https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0.pb"

BinaryClassifierID = "1y1A23D9wYJBEJ9XTv1Gd8qPiC8SOwVSM"
BinaryClassifierURL = 'https://drive.google.com/u/0/uc?id=1y1A23D9wYJBEJ9XTv1Gd8qPiC8SOwVSM'

def download(url, filename, message):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            print("Downloading ", message, ": ")
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format(
                    '█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


download(FrozenModelURL, "src/MegaDetector/cameratraps/detection/md_v4.1.0.pb", "Megadetector Frozen Model")
gdown.download(BinaryClassifierURL, "src/Binary_Model/PyTorch_Binary_Classifier.pth", quiet=False)