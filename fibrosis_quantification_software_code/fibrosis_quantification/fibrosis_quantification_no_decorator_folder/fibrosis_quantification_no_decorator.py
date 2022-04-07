# -*- coding: utf-8 -*-
import os
import typing

import keras
import numpy as np
import streamlit
import streamlit as st
from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR, INTER_AREA, cvtColor, resize
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage.util.shape import view_as_blocks
from tqdm import tqdm

from fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator_folder.helper_functions import (
    constrain_type,
    patchwise_threshold,
    set_global_mean,
    threshold_fx,
    threshold_gen,
    zip_up_image,
)


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
    patchwise_thresholded_tissue_nontissue = patchwise_threshold(img_array)
    # if radio == "WSI":
    #     st.image(patchwise_thresholded_tissue_nontissue)
    img_preprocess_blocks_255 = view_as_blocks(img_array, block_shape=(256, 256, 3)).squeeze()
    img_preprocess_blocks_255 = img_preprocess_blocks_255.reshape(-1, 256, 256, 3)
    # scale from [0,255] to [-1,1]
    im1_preprocess_blocks = (img_preprocess_blocks_255 - 127.5) / 127.5
    num_samples = im1_preprocess_blocks.shape[0]
    return (
        num_samples,
        im1_preprocess_blocks,
        img_preprocess_blocks_255,
        width,
        height,
        patchwise_thresholded_tissue_nontissue,
    )


def apply_gan(
    num_samples: int,
    model: keras.engine.functional.Functional,
    im1_preprocess_blocks: np.ndarray,
    img_preprocess_blocks_255: np.ndarray,
    width: int,
    height: int,
) -> typing.Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Call the model to translate the input"""
    assert type(num_samples) == int
    assert type(model) == keras.engine.functional.Functional
    assert type(im1_preprocess_blocks) == np.ndarray
    assert type(img_preprocess_blocks_255) == np.ndarray
    assert type(width) == int
    assert type(height) == int
    grid2d = []
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
        non_fibrotic_white, _, thresh_tissue = threshold_fx(patch_255)
        if non_fibrotic_white != 0:
            gen_calc, threshgen = threshold_gen(gen_image)
            grid2d.append("x")
        else:
            threshgen = np.full((thresh_tissue.shape), 255)
            grid2d.append("o")
        genner[sample] = gen_image
        threshgenner[sample] = threshgen

    zipped_genner = zip_up_image(height, width, genner)

    generated_image = 255 * zipped_genner
    generated_image = generated_image.astype(np.uint8)
    generated_image = cvtColor(generated_image, COLOR_BGR2RGB)
    st.markdown("<h5 style='text-align: center;'>AI generated translation</h1>", unsafe_allow_html=True)
    st.image(generated_image)

    # st.image(generated_thresholded)
    assert type(grid2d) == list
    assert type(threshgenner) == np.ndarray
    assert type(thresh_tissue) == np.ndarray
    return grid2d, threshgenner, thresh_tissue, generated_image


def clean_images(
    width: int, height: int, grid2d: list, threshgenner: np.ndarray, thresh_tissue: np.ndarray
) -> np.ndarray:
    """Clean the thresholded (fibrosis vs nonfibrosis) images"""
    assert type(width) == int
    assert type(height) == int
    assert type(grid2d) == list
    assert type(threshgenner) == np.ndarray
    assert type(thresh_tissue) == np.ndarray

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

    clean_thresholded_fibrosis_nonfibrosis = zip_up_image(height, width, threshgenner)
    clean_thresholded_fibrosis_nonfibrosis = clean_thresholded_fibrosis_nonfibrosis.astype(np.uint8)
    st.markdown("<h5 style='text-align: center;'>Fibrotic vs nonfibrotic pixels</h1>", unsafe_allow_html=True)
    st.image(clean_thresholded_fibrosis_nonfibrosis)
    assert type(clean_thresholded_fibrosis_nonfibrosis) == np.ndarray
    return clean_thresholded_fibrosis_nonfibrosis, remove


def report_fibrosis(patchwise_thresholded_tissue_nontissue, radio, clean_thresholded_fibrosis_nonfibrosis):
    """Report the amount of fibrosis"""
    assert type(patchwise_thresholded_tissue_nontissue) == np.ndarray
    assert type(radio) == str
    assert type(clean_thresholded_fibrosis_nonfibrosis) == np.ndarray
    final_thresholded_flat = clean_thresholded_fibrosis_nonfibrosis.flatten().tolist()
    fibrotic_pixels = final_thresholded_flat.count(0)
    non_fibrotic_white = final_thresholded_flat.count(255)
    total_pixels = len(final_thresholded_flat)
    assert fibrotic_pixels + non_fibrotic_white == total_pixels, "Major problem with math!"

    final_tissue_nontissue_flat = patchwise_thresholded_tissue_nontissue.flatten().tolist()
    total_pixels_tissue_nontissue = len(final_tissue_nontissue_flat)
    if radio == "WSI":
        print("Calculating for WSI")
        nontissue_black = final_tissue_nontissue_flat.count(0)
        tissue_white = final_tissue_nontissue_flat.count(255)
        assert nontissue_black + tissue_white == total_pixels_tissue_nontissue, "Major problem with math!"
        assert total_pixels_tissue_nontissue == total_pixels, "Pixel counts don't add up!"
    else:
        tissue_white = total_pixels_tissue_nontissue
        print("Calculating for patch")

    tissue_final = round(100 * tissue_white / total_pixels_tissue_nontissue, 2)
    fibrosis_final = round(100 * fibrotic_pixels / tissue_white, 2)
    st.write(
        f"The percentage of tissue in this image is {tissue_final}% and the percentage of fibrosis in this image is {fibrosis_final}%."
    )

    print(f"tissue white {tissue_white}")
    print(f"background black {total_pixels_tissue_nontissue}")
    print(f"fascia black {fibrotic_pixels}")
    assert type(tissue_final) == float
    assert type(fibrosis_final) == float
    return tissue_final, fibrosis_final
