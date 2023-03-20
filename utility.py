import numpy as np
import os
import cv2

def img_to_encoding(image_path):
    img1 = cv2.imread(image_path, -1)
    img = img1[...,::-1]
   
    return img

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(image_path, img)