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
    for i in range(len(mask_list)):
        mask_pad = np.pad(mask_list[i], 1, mode='edge')
        res = np.clip(mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 0)) + \
                    mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 0)) + \
                    mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 1)) + \
                    mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 1)), \
                    a_min = 0, a_max = 1)[1:-1, 1:-1]
        contour_list.append(res)
    return contour_list
    

def convert_contours_to_txt_lines(contour_list, class_list, width, height):
    lines = []
    for i, contour in enumerate(contour_list):
        line = []
        line.append(f"{class_list[i]}")
        mask_indices = np.where(contour == 1)
        for i in range(len(mask_indices[0])):
            x = mask_indices[0][i]
            y = mask_indices[1][i]
            line.append(f"{x/width}")
            line.append(f"{y/height}")
        line = ' '.join(line)
        lines.append(line)
    #append empty line at the end
    lines.append("\n")
    return lines

tik = time.time()

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
            cv2.imwrite(f"{mask_visualization_dir}{os.path.splitext(filename)[0]}_{i}.png", contour*255)
            print(f"{mask_visualization_dir}{os.path.splitext(filename)[0]}.png")

print(time.time()-tik)