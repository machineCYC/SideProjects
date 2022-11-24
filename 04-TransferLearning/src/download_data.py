import os, sys
import tarfile
import argparse
import pickle
import math
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "data")
CIFAR10_DATASET_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "cifar-10-batches-py")

SRC_TRAIN_DIR = os.path.join(DATA_FOLDER_PATH, "train")
DEST_TRAIN_DIR = os.path.join(DATA_FOLDER_PATH, "train2")


class _download_progress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)

        self.last_block = block_num


def download_original_data(data_folder_path, cifar10_dataset_folder_path):
    
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    """
        check if the data (gz) file is already downloaded
        if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz
    """
    if not os.path.isfile(os.path.join(data_folder_path, "cifar-10-python.tar.gz")):
        with _download_progress(unit="B", unit_scale=True, miniters=1, desc="CIFAR-10 Dataset") as pbar:
            urlretrieve(
                url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                filename=os.path.join(data_folder_path, "cifar-10-python.tar.gz"),
                reporthook=pbar.hook)

    if not os.path.isdir(cifar10_dataset_folder_path):
        with tarfile.open(os.path.join(data_folder_path, "cifar-10-python.tar.gz")) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=data_folder_path)
            tar.close()


# load raw data and reshape the input images
def _load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    
    if batch_id == "test":
        data_batch_name = "/test_batch"
    else:
        data_batch_name = "/data_batch_" + str(batch_id)
    
    with open(cifar10_dataset_folder_path + data_batch_name, mode="rb") as file:
        # note the encoding type is "latin1"
        batch = pickle.load(file, encoding="latin1")
        
    features = batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch["labels"]
        
    return features, labels


def _label_index_map_label_name(label_index):

    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    label_name = classes[label_index]

    return label_name


def _preprocess_and_save(features, labels, batch_i, data_folder_path):

    for i, (feature, label) in enumerate(zip(features, labels)):
        img = Image.fromarray(feature.astype(np.uint8))
        label_name = _label_index_map_label_name(label)
        image_name = "b" + str(batch_i) + "_" + label_name + "_" + str(i)
        print("Creat batch {} image with label: {} and file name: {}".format(batch_i, label_name, image_name))

        label_dir_path = os.path.join(data_folder_path, "train", label_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        img.save(os.path.join(label_dir_path, image_name + ".jpg"))

        label_dir_path = os.path.join(data_folder_path, "train2", label_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
        img.save(os.path.join(label_dir_path, image_name + ".jpg"))


def preprocess_and_save_image(cifar10_dataset_folder_path, data_folder_path):
    n_batches = 5

    for batch_i in range(1, n_batches + 1):
        features, labels = _load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # preprocess the whole training dataset of the batch
        # - Put each training image into a sub folder corresponding to its label after converting to JPG format
        # - named "b" + batch_number + label_name + image_index
        _preprocess_and_save(features, labels, batch_i, data_folder_path)
    
    # preprocess the whole testing dataset
    # - Put each training image into a sub folder corresponding to its label after converting to JPG format
    # - named "bt" + label_name + image_index
    features, labels = _load_cfar10_batch(cifar10_dataset_folder_path, "test")
    _preprocess_and_save(features, labels, "t", data_folder_path)


def generate_images_2_balance_classes(src_train_dir, dest_train_dir, class_size):
    
    # Some agile data augmentation (to prevent overfitting) + class balance
    datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode="nearest")
    
    for label_folder in os.listdir(src_train_dir):
        img_num = len([img_name for img_name in os.listdir(os.path.join(src_train_dir, label_folder)) if img_name.endwith(".jpg")])
        #nb of generations per image for this class label in order to make it size ~= class_size
        ratio = math.floor(class_size / img_num) - 1
        print("label {} have {} imgs, generate {} imgs".format(label_folder, img_num, img_num * (ratio + 1)))

        dest_lab_dir = os.path.join(dest_train_dir, label_folder)
        src_lab_dir = os.path.join(src_train_dir, label_folder)
        if not os.path.exists(dest_lab_dir):
            os.makedirs(dest_lab_dir)

        for img_path in os.listdir(src_lab_dir):
            img = load_img(os.path.join(src_lab_dir, img_path))
            #img.save(os.path.join(dest_lab_dir,file))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=dest_lab_dir, save_format="jpg"):
                i += 1
                if i > ratio:
                    break


def calculate_image_number(cal_dir):

    base_name = os.path.basename(cal_dir)
    for dirpath, dirnames, filenames in os.walk(cal_dir):
        i = 0
        label = ""
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            label = os.path.split(dirpath)[1]
            i += 1
        print(base_name, label, i)


def main(args):
    download_original_data(args.data_folder_path, args.cifar10_dataset_folder_path)

    preprocess_and_save_image(args.cifar10_dataset_folder_path, args.data_folder_path)

    calculate_image_number(args.src_train_dir)

    # generate_images_2_balance_classes(args.src_train_dir, args.dest_train_dir, args.class_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "--data_folder_path",
      type=str,
      default=DATA_FOLDER_PATH,
      help="Path to folders of data"
    )
    parser.add_argument(
      "--cifar10_dataset_folder_path",
      type=str,
      default=CIFAR10_DATASET_FOLDER_PATH,
      help="Path to folders of cifar10 data"
    )
    parser.add_argument(
      "--src_train_dir",
      type=str,
      default=SRC_TRAIN_DIR,
      help="Path to folder of creating training folder, named 'train'. The path is created automatic in project folder/data/train"
    )
    parser.add_argument(
      "--dest_train_dir",
      type=str,
      default=DEST_TRAIN_DIR,
      help="Path to folder of creating training folder, named 'train2'. The path is created automatic in project folder/data/train2 "
    )
    parser.add_argument(
      "--class_size",
      type=int,
      default=10000,
      help="Data size of each class"
    )
    main(parser.parse_args())