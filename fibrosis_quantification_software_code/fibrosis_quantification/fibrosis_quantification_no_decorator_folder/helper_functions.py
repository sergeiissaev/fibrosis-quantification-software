# -*- coding: utf-8 -*-
import typing

import cv2
import numpy as np
from cv2 import hconcat, vconcat
from tqdm import tqdm


def callback_colors(image):
    image = image.flatten().tolist()
    print(f"White: {image.count(255)}")
    print(f"Black: {image.count(0)}")


def constrain_type(patch: np.ndarray) -> np.ndarray:
    assert type(patch) == np.ndarray
    patch = np.clip(patch, 0, 255)
    patch = patch.astype(np.uint8)
    return patch


def constrain_type_backwards(patch: np.ndarray) -> np.ndarray:
    assert type(patch) == np.ndarray
    patch = patch.astype(np.uint8)
    patch = np.clip(patch, 0, 255)
    return patch


def set_global_mean(patch: np.ndarray, desired_mean: int) -> np.ndarray:
    assert type(patch) == np.ndarray
    add = int(desired_mean - patch.mean())
    patch = patch + add
    return patch


def threshold_fx(patch: np.ndarray) -> typing.Tuple[int, int, np.ndarray]:
    assert type(patch) == np.ndarray
    # Get the mean and standard deviation of the desired patch
    patch = patch.squeeze()
    patch_mean_original = patch.mean()
    patch_std_original = np.std(patch)
    # Set the global mean of the patch to 100
    patch = set_global_mean(patch, 100)
    patch = constrain_type(patch)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = ~patch
    patch = 2 * patch
    patch = constrain_type(patch)
    patch = cv2.medianBlur(patch, 7)
    patch = set_global_mean(patch, 100)
    patch = constrain_type(patch)
    ranger = abs(int(patch.min()) - int(patch.max()))
    # Select threshold value based on mean and standard deviation of result.
    if patch_std_original < 4.5:
        thresh_calc = 254
    elif patch_mean_original > 200:
        thresh_calc = patch.min() + (0.075 * ranger)
    elif ranger < 50:
        thresh_calc = patch.min() + (0.15 * ranger)
    elif patch_std_original < 8:
        thresh_calc = patch.min() + (0.085 * ranger)
    else:
        thresh_calc = patch.min() + (0.06 * ranger)
    ret, thresh1 = cv2.threshold(patch, thresh_calc, 255, cv2.THRESH_BINARY)
    if np.std(thresh1) != 0:
        white = 0
        black = 0
        for column in thresh1:
            for pixel in column:
                if pixel == 255:
                    white += 1
                elif pixel == 0:
                    black += 1
    else:
        white = 0
        black = 65536
    assert white + black == 256 * 256
    assert type(white) == int
    assert type(black) == int
    assert type(thresh1) == np.ndarray
    return white, black, thresh1


def patchwise_threshold(patch: np.ndarray):
    """Obtain the image-wise threshold for an input WSI"""
    assert type(patch) == np.ndarray
    callback_colors(patch)
    # Set the global mean of the patch to 100
    patch = set_global_mean(patch, 100)
    patch = constrain_type_backwards(patch)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = ~patch
    patch = 2 * patch
    patch = constrain_type_backwards(patch)
    patch = set_global_mean(patch, 100)
    patch = constrain_type_backwards(patch)
    patch = cv2.medianBlur(patch, 15)
    kernel = np.ones((5, 5), np.uint8)
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    patch = set_global_mean(patch, 100)
    patch = constrain_type_backwards(patch)
    ranger = abs(int(patch.min()) - int(patch.max()))
    thresh_calc = patch.min() + (0.20 * ranger)
    if patch.min() < 50:
        thresh_calc = patch.min() + (0.40 * ranger)
    if patch.min() < 30:
        thresh_calc = patch.min() + (0.50 * ranger)
    _, thresh1 = cv2.threshold(patch, thresh_calc, 255, cv2.THRESH_BINARY)

    callback_colors(thresh1)
    return thresh1


def threshold_gen(gen_image: np.ndarray) -> typing.Tuple[int, np.ndarray]:
    norm_image = cv2.normalize(gen_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    pic = norm_image.squeeze()
    img = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    inv = ~img
    blimg = cv2.medianBlur(inv, 5)
    ranger = abs(int(blimg.min()) - int(blimg.max()))
    thresh_calc = int(blimg.min() + (0.93 * ranger))
    ret, threshgen = cv2.threshold(blimg, thresh_calc, 255, cv2.THRESH_BINARY)
    black_fib = 0
    for column in threshgen:
        for pixel in column:
            if pixel == 0:
                black_fib += 1
    assert type(black_fib) == int
    assert type(threshgen) == np.ndarray
    return black_fib, threshgen


def zip_up_image(height, width, array_of_patches):
    multiple = 0
    for a in tqdm(range(height)):
        for k in range(width):
            if k == 0:
                im_h = array_of_patches[k + (width * multiple)]
            else:
                im_h = hconcat([im_h, array_of_patches[k + (width * multiple)]])
        multiple += 1
        if a == 0:
            generated_image = im_h
        else:
            generated_image = vconcat([generated_image, im_h])
    return generated_image
