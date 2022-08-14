import os
import torch


def calculate_statistic(img_dir):
    means = torch.zeros(3)
    stds = torch.zeros(3)

    images_list = os.listdir(img_dir)

    for im in images_list:
        full = os.path.join(img_dir, im)
        means += torch.mean(full, dim=(1,2))
        stds += torch.std(full, dim=(1,2))
   
    means /= len(images_list)
    stds /= len(images_list)

    return means, stds


def min_max_normalize(image):
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-5)
    return image


def depth_normalize(image, bit_depth=8):
    image = image / (2 ** bit_depth - 1)
    return image