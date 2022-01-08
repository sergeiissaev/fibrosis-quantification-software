# -*- coding: utf-8 -*-
import os

import numpy as np
import streamlit
import streamlit as st
from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR, INTER_AREA, cvtColor, hconcat, resize, vconcat
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage.util.shape import view_as_blocks
from tqdm import tqdm

from fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator.helper_functions import (
    constrain_type,
    patchwise_threshold,
    set_global_mean,
    threshold_fx,
    threshold_gen,
)


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


def import_model():
    model_file = os.path.join("models", "model_187000.h5")
    # load model
    model = load_model(model_file, compile=False)
    return model


def preliminary_preprocessing(source_file: streamlit.uploaded_file_manager.UploadedFile, radio):
    image = Image.open(source_file)
    img_array = np.array(image)
    img_array = cvtColor(img_array, COLOR_RGB2BGR)
    (h1, w1, d1) = img_array.shape
    width = w1 // 256
    height = h1 // 256
    dim = (width * 256, height * 256)
    img_array = resize(img_array, dim, interpolation=INTER_AREA)
    img_array = set_global_mean(img_array, 120)
    img_array = constrain_type(img_array)
    assert type(img_array) == np.ndarray
    if radio == "WSI":
        patchwise_thresholded = patchwise_threshold(img_array)
        st.image(patchwise_thresholded)
    img_preprocess_blocks_255 = view_as_blocks(img_array, block_shape=(256, 256, 3)).squeeze()
    img_preprocess_blocks_255 = img_preprocess_blocks_255.reshape(-1, 256, 256, 3)
    # scale from [0,255] to [-1,1]
    im1_preprocess_blocks = (img_preprocess_blocks_255 - 127.5) / 127.5
    num_samples = im1_preprocess_blocks.shape[0]
    return num_samples, im1_preprocess_blocks, img_preprocess_blocks_255, width, height


def apply_gan(num_samples: int, model, im1_preprocess_blocks, img_preprocess_blocks_255, width, height):
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
        invariant_datagen = ImageDataGenerator()
        invariant_datagen.fit(patch_1, augment=True, seed=12)
        invariant_generator = invariant_datagen.flow(patch_1, seed=12)
        invariant_generator.next()
        invariant = invariant_generator[0].copy()
        gen_image = model.predict(invariant)
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

    zipped_genner = zip_up_image(height, width, genner)
    zipped_threshgenner = zip_up_image(height, width, threshgenner)
    legacy_threshold = zip_up_image(height, width, thresher)

    generated_image = 255 * zipped_genner
    generated_image = generated_image.astype(np.uint8)
    generated_image = cvtColor(generated_image, COLOR_BGR2RGB)
    generated_thresholded = zipped_threshgenner.astype(np.uint8)
    legacy_threshold = legacy_threshold.astype(np.uint8)
    st.image(legacy_threshold)
    st.image(generated_thresholded)
    st.image(generated_image)

    supergrid = list()
    k = 0
    multiple = 0
    while k < height:
        supergrid.append(grid2d[(width * multiple) : ((multiple + 1) * width)])
        multiple += 1
        k += 1
    x = 0
    remove = list()
    width = len(supergrid[0])
    while x < len(supergrid):
        report = []
        for k in tqdm(range(width)):
            occupied = 0
            if supergrid[x][k] == "x":
                try:
                    if supergrid[x][k + 1] == "x":
                        occupied += 1
                except Exception as e:
                    print(e)
                try:
                    if supergrid[x][k - 1] == "x":
                        occupied += 1
                except Exception as e:
                    print(e)
                if occupied < 1:
                    # print('uh oh at position', x, k ,' -> only', occupied, 'are occupied!')
                    # If not first line
                    if x != 0:
                        try:
                            if supergrid[x - 1][k - 1] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                        try:
                            if supergrid[x - 1][k] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                        try:
                            if supergrid[x - 1][k + 1] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                    # If not last line
                    if x != len(supergrid):
                        try:
                            if supergrid[x + 1][k - 1] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                        try:
                            if supergrid[x + 1][k] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                        try:
                            if supergrid[x + 1][k + 1] == "x":
                                occupied += 1
                        except Exception as e:
                            print(e)
                    if occupied < 2:
                        # print('problem! occupied is ', occupied, 'location', k + (x * width), 'REMOVED!!')
                        remove.append(k + (x * width))
            report.append(occupied)
        x += 1

    for k in range(len(threshgenner)):
        if threshgenner[k].mean() == 0:
            remove.append(k)

    for i in remove:
        threshgenner[i] = np.full((thresh_tissue.shape), 255)

    clean_thresholded = zip_up_image(height, width, threshgenner)
    clean_thresholded = clean_thresholded.astype(np.uint8)
    st.image(clean_thresholded)

    return None
