# -*- coding: utf-8 -*-
from random import randint

import numpy as np
import streamlit as st
from cv2 import hconcat, vconcat
from skimage.util.shape import view_as_blocks
from tqdm import tqdm

from fibrosis_quantification_software_code.fibrosis_quantification.fibrosis_quantification_no_decorator_folder.helper_functions import (
    zip_up_image,
)


def callback_colors(image):
    image = image.flatten().tolist()
    print(f"White: {image.count(255)}")
    print(f"Black: {image.count(0)}")


def blob_removal(radio: str, patchwise_thresholded_tissue_nontissue, remove, samples, height, width) -> np.ndarray:
    """Call the model to translate the input"""
    assert type(radio) == str
    callback_colors(patchwise_thresholded_tissue_nontissue)
    if radio == "WSI":
        print("detected WSI")
        thresh_blocks = view_as_blocks(patchwise_thresholded_tissue_nontissue, block_shape=(256, 256)).squeeze()
        thresh_blocks = thresh_blocks.reshape(-1, 256, 256)
        thresh_beauty = np.zeros((samples, 256, 256))
        for i in range(samples):
            if i not in remove:
                thresh_beauty[i] = thresh_blocks[i]
        thresh_beauty_im = zip_up_image(height, width, thresh_beauty)
        thresh_beauty_im = thresh_beauty_im.astype(np.uint8)
        thresh_beauty = view_as_blocks(thresh_beauty_im, block_shape=(128, 128)).squeeze()
        thresh_beauty = thresh_beauty.reshape(-1, 128, 128)

        blobs = []
        for layer in tqdm(range(height * 2)):
            for i in range(width * 2 * layer, width * 2 * (layer + 1)):
                if thresh_beauty[i].mean() != 0:
                    blob = True
                    for l_patch in thresh_beauty[i][0]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in thresh_beauty[i][-1]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in range(len(thresh_beauty[i])):
                        if thresh_beauty[i][l_patch][0] != 0:
                            blob = False
                        if thresh_beauty[i][l_patch][-1] != 0:
                            blob = False
                    if blob:
                        blobs.append(i)
        for i in blobs:
            thresh_beauty[i] = np.full((128, 128), 0)
        multiple = 0
        for a in range(height * 2):
            for k in range(width * 2):
                if k == 0:
                    im_h = thresh_beauty[k + (width * 2 * multiple)]
                else:
                    im_h = hconcat([im_h, thresh_beauty[k + (width * 2 * multiple)]])
            multiple += 1
            if a == 0:
                thresh_beauty_im = im_h
            else:
                thresh_beauty_im = vconcat([thresh_beauty_im, im_h])
        thresh_beauty_im = thresh_beauty_im.astype(np.uint8)

        thresh_beauty = view_as_blocks(thresh_beauty_im, block_shape=(128, 256)).squeeze()
        thresh_beauty = thresh_beauty.reshape(-1, 128, 256)

        # In[74]:

        blobs = []
        for layer in tqdm(range(height * 2)):
            for i in range(width * layer, width * (layer + 1)):
                if thresh_beauty[i].mean() != 0:
                    blob = True
                    for l_patch in thresh_beauty[i][0]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in thresh_beauty[i][-1]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in range(len(thresh_beauty[i])):
                        if thresh_beauty[i][l_patch][0] != 0:
                            blob = False
                        if thresh_beauty[i][l_patch][-1] != 0:
                            blob = False
                    if blob:
                        blobs.append(i)
                        # print('OMG BLOB')

        # In[75]:

        for i in blobs:
            thresh_beauty[i] = np.full((128, 256), 0)
        multiple = 0
        for a in range(height * 2):
            for k in range(width):
                if k == 0:
                    im_h = thresh_beauty[k + (width * multiple)]
                else:
                    im_h = hconcat([im_h, thresh_beauty[k + (width * multiple)]])

            multiple += 1
            if a == 0:
                thresh_beauty_im = im_h
            else:
                thresh_beauty_im = vconcat([thresh_beauty_im, im_h])
        thresh_beauty_im = thresh_beauty_im.astype(np.uint8)

        thresh_beauty = view_as_blocks(thresh_beauty_im, block_shape=(256, 128)).squeeze()
        thresh_beauty = thresh_beauty.reshape(-1, 256, 128)

        # In[78]:

        blobs = []
        for layer in tqdm(range(height)):
            for i in range(width * 2 * layer, width * 2 * (layer + 1)):
                if thresh_beauty[i].mean() != 0:
                    blob = True
                    for l_patch in thresh_beauty[i][0]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in thresh_beauty[i][-1]:
                        if l_patch != 0:
                            blob = False
                    for l_patch in range(len(thresh_beauty[i])):
                        if thresh_beauty[i][l_patch][0] != 0:
                            blob = False
                        if thresh_beauty[i][l_patch][-1] != 0:
                            blob = False
                    if blob:
                        blobs.append(i)
                        # print('OMG BLOB')

        # In[79]:

        for i in blobs:
            thresh_beauty[i] = np.full((256, 128), 0)

        # In[80]:

        multiple = 0
        for a in range(height):
            for k in range(width * 2):
                if k == 0:
                    im_h = thresh_beauty[k + (width * 2 * multiple)]
                else:
                    im_h = hconcat([im_h, thresh_beauty[k + (width * 2 * multiple)]])
            multiple += 1
            if a == 0:
                thresh_beauty_im = im_h
            else:
                thresh_beauty_im = vconcat([thresh_beauty_im, im_h])
        thresh_beauty_im = thresh_beauty_im.astype(np.uint8)

        for _ in tqdm(range(10000)):
            x = randint(0, thresh_beauty_im.shape[0] - 64)
            y = randint(0, thresh_beauty_im.shape[1] - 64)

            random_patch = thresh_beauty_im[x : x + 64, y : y + 64]

            if random_patch.mean() != 0:
                blob = True
                for l_patch in random_patch[0]:
                    if l_patch != 0:
                        blob = False
                for l_patch in random_patch[-1]:
                    if l_patch != 0:
                        blob = False
                for l_patch in range(len(random_patch)):
                    if random_patch[l_patch][0] != 0:
                        blob = False
                    if random_patch[l_patch][-1] != 0:
                        blob = False
                if blob:
                    blobs.append(i)
                    # print('OMG BLOB')

                    thresh_beauty_im[x : x + 64, y : y + 64] = np.full((64, 64), 0)

        thresh_beauty_im = thresh_beauty_im.astype(np.uint8)
        st.markdown("<h5 style='text-align: center;'>Final Tissue Detection Image</h1>", unsafe_allow_html=True)
        st.image(thresh_beauty_im)

        return thresh_beauty_im
    else:
        return patchwise_thresholded_tissue_nontissue
