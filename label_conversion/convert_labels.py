import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_seg_txt(label_path, height, width):
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


label_path = "original_labels/green_h_image1.txt"
"""

contour_list = []
for i in range(len(mask_list)):
    mask_pad = np.pad(mask_list[i], 1, mode='edge')
    res = np.clip(mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 0)) + \
                mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 0)) + \
                mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 1)) + \
                mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 1)), \
                a_min = 0, a_max = 1)[1:-1, 1:-1]
    contour_list.append(res)

# convert the contours back to the original COCO/Yolo format
cv2.imwrite("masks_viz/mask0.png", contour_list[0])
plt.imshow(contour_list[0])

lines = []

for i, contour in enumerate(contour_list):
    line = []
    line.append(f"{class_list[i]}")
    mask_indices = np.where(contour == 1)
    for i in range(len(mask_indices[0])):
        x = mask_indices[0][i]
        y = mask_indices[1][i]
        line.append(f"{x/width}")
        line.append(f"{y/width}")
    
    line = ' '.join(line)
    lines.append(line)

with open('new_labels/green_h_image1.txt', 'w') as f:
    f.writelines(lines)
"""

