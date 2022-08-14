"""Script for cutting large images to patches. Also script make dataset directories and split set to test/train/val subsets"""
import os
import numpy as np
from skimage import io
from typing import List
from glob import glob
import shutil
from sklearn.model_selection import train_test_split
import datetime


ROOT_DIR = "/home/n.kotov1/Dataset_bpla_filtered"
DATASET_DIR = "/raid/n.kotov1/Dataset_bpla_filtered"
PATCHES_DIR = "/raid/n.kotov1/Dataset_bpla_patches"

PATCH_SIZE = 512
STEP = 256


def extract_classes_names(root_dir: str) -> List[str]:
    """ Extract classses names from directories """
    classes_names = []

    for folder in os.listdir(root_dir):
        full_path = os.path.join(root_dir, folder)
        if os.path.isdir(full_path):
            cls_name = full_path.split('/')[-1]
            classes_names.append(cls_name)
    
    return classes_names


def make_tree_directories(input_dir: str, output_dir: str) -> None:
    """ Make tree directories in format <train/test/val>/<classes> """
    train_test_val = ["train", "test", "val"]
    classes = extract_classes_names(input_dir)

    for folder in train_test_val:
        for clss in classes:
            full = os.path.join(output_dir, folder, clss)
            os.makedirs(full)


def train_test_val_split(root_dir: str, out_dir: str) -> None:
    """ Function split all images to train/test/val subsets and copy images there """
    dest_folders = ["train", "test", "val"]

    for clss in os.listdir(root_dir):
        root_cls = os.path.join(root_dir, clss)
        if os.path.isdir(root_cls):
            full_path = glob(os.path.join(root_cls, '*.JPG'), recursive=True)

            im_train_val, im_test = train_test_split(full_path, test_size=0.2, shuffle=True)
            im_train, im_val = train_test_split(im_train_val, test_size=0.1, shuffle=True)

            for subset, ds in zip([im_train, im_test, im_val], dest_folders):
                for im in subset:
                    im_cl = im.split("/")[-2]
                    destination = os.path.join(out_dir, ds, im_cl)
                    shutil.copy(im, destination)
                    print(f"[INFO][{datetime.datetime.now().strftime('%H:%M:%S')}]: '{im}' has been copied to '{destination}'")


def extract_patches(im_dir: str, out_dir: str) -> None:
    """ Cutting large images to rectangle patches """
    list_large_im = glob(os.path.join(im_dir, "*", "*", "*.JPG"), recursive=True)

    for im in list_large_im:
        img = io.imread(im)

        for x in np.arange(0, img.shape[1] - PATCH_SIZE + 1, STEP):
            for y in np.arange(0, img.shape[0] - PATCH_SIZE + 1, STEP):
                name, ext = os.path.splitext(im)
                im_slice = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                destination = os.path.join(out_dir, os.path.join(im.split("/")[4], im.split("/")[5]), f"{name.split('/')[-1].split('.')[0]}_{x}_{y}{ext}")
                io.imsave(destination, im_slice)
                print(f"[INFO][{datetime.datetime.now().strftime('%H:%M:%S')}]: patch has been saved to '{destination}'")


def main():
    """ Main function """
    make_tree_directories(ROOT_DIR, DATASET_DIR)
    train_test_val_split(ROOT_DIR, DATASET_DIR)
    make_tree_directories(ROOT_DIR, PATCHES_DIR)
    extract_patches(DATASET_DIR, PATCHES_DIR)
    shutil.rmtree(DATASET_DIR)


if __name__ == "__main__":
    main()
