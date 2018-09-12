import os
import zipfile
from tqdm import tqdm
from urllib.request import urlretrieve


PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "data")
DATASET_ZIP_PATH = os.path.join(DATA_FOLDER_PATH, "a0409a00-8-dataset_dp.zip")

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)

        self.last_block = block_num

if not os.path.isdir(DATA_FOLDER_PATH):
    os.makedirs(DATA_FOLDER_PATH)

""" 
    check if the data (zip) file is already downloaded
    if not, download it from "https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip
"""
if not os.path.isfile(os.path.join(DATA_FOLDER_PATH, "a0409a00-8-dataset_dp.zip")):
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc="hackerearth challenge-1 datasets") as pbar:
        urlretrieve(
            url="https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip",
            filename=os.path.join(DATA_FOLDER_PATH, "a0409a00-8-dataset_dp.zip"),
            reporthook=pbar.hook)

if os.path.isfile(DATASET_ZIP_PATH):
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zf:
        zf.extractall(path=DATA_FOLDER_PATH)
        zf.close()
