import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time


def parse_seg_txt(label_path, width, height):
    with open(label_path,"r") as f:
        lines = f.readlines()

    masks = []
    classes = []

    for line in lines:
        line_arr = line.split()
        if line_arr:  # make sure line is not empty
            classes.append(int(line_arr[0]))
            mask = np.zeros((height, width))
            for i in range(1, len(line_arr), 2):
                x, y = round(float(line_arr[i])*(width-1)), round(float(line_arr[i+1])*(height-1))
                mask[y, x] = 1
            masks.append(mask)
    return classes, masks


def get_contours_from_masks(mask_list):
    contour_list = []
    for mask in mask_list:
        ret, thresh = cv2.threshold(mask, 0.5, 1, 0)
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_list.append(contours[0])
    return contour_list
    

def convert_contours_to_txt_lines(contour_list, class_list, width, height):
    lines = []
    for i, contour in enumerate(contour_list):
        line = []
        line.append(f"{class_list[i]}")
        for point in contour:
            line.append(f"{point[0][0]/width}")
            line.append(f"{point[0][1]/height}")
        line = ' '.join(line)
        lines.append(line)
    return lines


width, height = 1920, 1080

label_in_dir = "original_labels/"
label_out_dir = "new_labels/"
mask_visualization_dir = "masks_viz/"

# loop through all .txt files in the directory
for filename in os.listdir(label_in_dir):
    if filename.endswith(".txt"):
        label_path = label_in_dir + filename
        class_list, mask_list = parse_seg_txt(label_path, width, height)
        contour_list = get_contours_from_masks(mask_list)
        lines = convert_contours_to_txt_lines(contour_list, class_list, width, height)
        with open(label_out_dir + filename, "w") as f:
            f.writelines('\n'.join(lines))
        # visualize the masks
        for i, contour in enumerate(contour_list):
            mask = np.zeros((height, width))
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), 1)
            cv2.imwrite(f"{mask_visualization_dir}{os.path.splitext(filename)[0]}_{i}.png", mask)
            print(f"{mask_visualization_dir}{os.path.splitext(filename)[0]}.png")
