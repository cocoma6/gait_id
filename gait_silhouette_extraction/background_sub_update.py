"""
background_sub_update.ipynb

Original file is located at
    https://colab.research.google.com/drive/1Lh7zRBH8gpQ0r0_zRHyfVERWHCrFw4E5

# Data

# connet kaggle
!pip install kaggle
!mkdir .kaggle

import json
token = {"username":"kemapow","key":"169f15df5f9fb93dfc175cf6433799c6"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}

# load data
!kaggle datasets download -d saimadhurivasam/human-activity-recognition-from-video -p /content/
!unzip -d /content/ /content/*.zip
"""

# Perform background subtraction

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

def image_enhancement(frame):
    err_kernel = np.ones((5, 5), np.uint8)
    dil_kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(frame, err_kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, dil_kernel, iterations=1)
    _, img_threshold = cv2.threshold(img_dilation, 100, 255, cv2.THRESH_BINARY)
    return img_threshold

def set_new_repo(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("new folder: %s" %path)

'''
source: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
'''
def background_subtract(path, sub_name):
    OUTPUT_PATH = '/content/Output/' # set the real output path
    full_path = OUTPUT_PATH + sub_name
    set_new_repo(full_path)

    backSub = cv2.createBackgroundSubtractorKNN(detectShadows = False)
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(path))
    if not capture.isOpened():
        print('Unable to open: ' + path)
        exit(0)

    while True:
        ret, frame = capture.read()
        idx = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame is None:
            break
        
        fgMask1 = backSub.apply(frame) # background subtraction
        fgMask2 = cv2.medianBlur(fgMask1, 3) # median filter
        fgMask = image_enhancement(fgMask2) # image morphology

        if np.count_nonzero(fgMask) > 1000:
            plt.imsave("%s/silho_%d.jpg"  %(full_path, idx), fgMask)
            # plot the result
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(1,3,1)  
            ax1.imshow(frame)
            ax1.set_title("frame: %d"%idx)
            ax2 = fig.add_subplot(1,3,2)  
            ax2.imshow(fgMask1, cmap='gray')
            ax2.set_title("background substraction")
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(fgMask, cmap='gray')
            ax3.set_title("silhouette")
            plt.savefig("%s/result_%d.jpg" %(full_path, idx))
            plt.show()
            
        else:
            print("Working on frame %d, no silhouette" %idx)


# main

INPUT_PATH = '/content/Data/walking/' # set the real input path
id_list = os.listdir(INPUT_PATH)
id_list.sort()
# Walk the input path
for _id in id_list:
    path = os.path.join(INPUT_PATH, _id)
    background_subtract(path, _id)