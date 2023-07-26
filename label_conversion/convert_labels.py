import numpy as np
import cv2

with open("original_labels/green_h_image1.txt","r") as f:
    lines = f.readlines()

height, width = 1080, 1920  # replace with your actual image dimensions
mask_list = []
class_list = []

for line in lines:
    line_arr = line.split()
    if line_arr:  # make sure line is not empty
        class_list.append(int(line_arr[0]))
        mask = np.zeros((height, width))
        for i in range(1, len(line_arr), 2):
            x = int(float(line_arr[i])*width) -1
            y = int(float(line_arr[i])*height) -1
            mask[y, x] = 1
    mask_list.append(mask)