import numpy as np
import cv2
import os
import time

def parse_seg_txt(label_path, width, height):
    data = np.loadtxt(label_path)
    classes = data[:, 0].astype(int)
    coords = data[:, 1:].reshape(-1, 2) * ([width - 1, height - 1])
    coords = np.round(coords).astype(int)
    masks = np.zeros((len(classes), height, width))
    masks[np.arange(len(classes)), coords[:, 1], coords[:, 0]] = 1
    return classes, masks


def get_contours_from_masks(mask_list):
    mask_pad = np.pad(mask_list, ((0, 0), (1, 1), (1, 1)), mode='edge')
    contour_list = np.clip(mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 1)) + \
                        mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 1)) + \
                        mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, 1, 2)) + \
                        mask_pad - cv2.bitwise_and(mask_pad, np.roll(mask_pad, -1, 2)), \
                        a_min = 0, a_max = 1)[:, 1:-1, 1:-1]
    return contour_list


def convert_contours_to_txt_lines(contour_list, class_list, width, height):
    mask_indices = np.column_stack(np.where(contour_list == 1))
    mask_indices[:, [1, 2]] /= [width, height]
    lines = np.column_stack((class_list[mask_indices[:, 0]], mask_indices[:, 1:]))
    return [' '.join(map(str, line)) for line in lines]

tik = time.time()

width, height = 1920, 1080

label_in_dir = "original_labels/"
label_out_dir = "new_labels/"
mask_visualization_dir = "masks_viz/"

# loop through all .txt files in the directory
for filename in os.listdir(label_in_dir):
    if filename.endswith(".txt"):
        label_path = os.path.join(label_in_dir, filename)
        class_list, mask_list = parse_seg_txt(label_path, width, height)
        contour_list = get_contours_from_masks(mask_list)
        lines = convert_contours_to_txt_lines(contour_list, class_list, width, height)
        # write lines to the output file
        with open(os.path.join(label_out_dir, filename), "w") as f:
            f.write('\n'.join(lines))
        # visualize the masks
        for i in range(contour_list.shape[0]):
            cv2.imwrite(f"{mask_visualization_dir}{os.path.splitext(filename)[0]}_{i}.png", contour_list[i]*255)

print(time.time()-tik)
