# -*- coding: utf-8 -*-
import os
import typing

import cv2
import numpy as np
import streamlit
from cv2 import INTER_AREA, resize
from keras.models import load_model
from PIL import Image
from skimage.util.shape import view_as_blocks
from tqdm import tqdm


def constrain_type(patch: np.ndarray) -> np.ndarray:
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


def import_model():
    model_file = os.path.join("models", "model_187000.h5")
    # load model
    model = load_model(model_file, compile=False)
    return model


def preliminary_preprocessing(source_file: streamlit.uploaded_file_manager.UploadedFile):
    image = Image.open(source_file)
    img_array = np.array(image)
    (h1, w1, d1) = img_array.shape
    width = w1 // 256
    height = h1 // 256
    dim = (width * 256, height * 256)
    img_array = resize(img_array, dim, interpolation=INTER_AREA)
    img_array = set_global_mean(img_array, 120)
    img_array = constrain_type(img_array)
    assert type(img_array) == np.ndarray
    img_preprocess_blocks_255 = view_as_blocks(img_array, block_shape=(256, 256, 3)).squeeze()
    img_preprocess_blocks_255 = img_preprocess_blocks_255.reshape(-1, 256, 256, 3)
    # scale from [0,255] to [-1,1]
    im1_preprocess_blocks = (img_preprocess_blocks_255 - 127.5) / 127.5
    num_samples = im1_preprocess_blocks.shape[0]
    return num_samples, im1_preprocess_blocks, img_preprocess_blocks_255


def apply_gan(num_samples: int, model, im1_preprocess_blocks, img_preprocess_blocks_255):
    grid2d = []
    thresher = np.zeros((num_samples, 256, 256))
    genner = np.zeros((num_samples, 256, 256, 3))
    threshgenner = np.zeros((len(im1_preprocess_blocks), 256, 256))
    for sample in tqdm(range(num_samples)):
        patch_1 = im1_preprocess_blocks[[sample]]
        patch_255 = img_preprocess_blocks_255.astype(np.uint8)[sample]
        assert len(patch_1.shape) == 4
        assert type(patch_1) == np.ndarray
        assert type(patch_255) == np.ndarray
        gen_image = model.predict(patch_1)
        gen_image = (gen_image + abs(gen_image.min())) / 2.0
        assert len(gen_image.shape) == 4
        assert type(gen_image) == np.ndarray
        white, _, thresh_tissue = threshold_fx(patch_255)
        if white != 0:
            gen_calc, threshgen = threshold_gen(gen_image)
            grid2d.append("x")
        else:
            threshgen = np.full((thresh_tissue.shape), 255)
            grid2d.append("o")
        thresh_tissue = np.expand_dims(thresh_tissue, axis=0)
        genner[sample] = gen_image
        thresher[sample] = thresh_tissue
        threshgenner[sample] = threshgen

    return None
