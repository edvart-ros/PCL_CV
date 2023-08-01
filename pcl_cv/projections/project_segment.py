import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time


# create empty image
height, width, channels = 1080, 1920, 3
min_range, max_range = 8.0, 10.0
projection_img = np.zeros((height, width, channels), dtype=np.float32)
num_pixels = 1000000

# generate random pixels and corresponding points data
projected_pixels = np.random.randint(0, [height, width], (num_pixels, 2))
projected_points = np.random.uniform(min_range, max_range, [num_pixels, 3])
# highlight these pixels for visualization
projection_img[projected_pixels[:,0], projected_pixels[:,1]] = projected_points

# create the binary polygon mask (single-channel)
poly_mask = np.zeros((height, width), dtype=np.uint8)
polygon_corners = np.array([(900, 200), (1100, 350), (800, 1000), (500, 600)]).reshape((-1, 1, 2))
cv2.fillPoly(poly_mask, [polygon_corners], color=1)

# get the indices of the projected pixels/points which lie on the mask
returns_indices = np.nonzero(poly_mask[projected_pixels[:, 0], projected_pixels[:, 1]])
pixels = projected_pixels[returns_indices]
points = projected_points[returns_indices]
point_estimate = np.median(points, axis=0)


masked_projections = np.zeros([1080, 1920, 3])
masked_projections[pixels[:,0], pixels[:,1],:] = points


fig, axs = plt.subplots(3, 1, figsize=(10, 10))

axs[0].imshow(projection_img / max_range, cmap='jet')
axs[0].set_title("Original Image")

axs[1].imshow(poly_mask, cmap='gray')
axs[1].set_title("Mask")

axs[2].imshow(masked_projections / max_range, cmap='jet')
axs[2].set_title("Masked Image")

plt.savefig("segment_out.svg")