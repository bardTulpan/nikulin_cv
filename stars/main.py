import numpy as np
from skimage import morphology
from skimage.measure import label as sk_label

img = np.load('stars.npy')

cross_struct = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
])

x_struct = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]
])

cross_filtered = morphology.opening(img, footprint=cross_struct)
x_filtered = morphology.opening(img, footprint=x_struct)

cross_labels = sk_label(cross_filtered)
x_labels = sk_label(x_filtered)

total_objects = cross_labels.max() + x_labels.max()
print(f"Total detected objects: {total_objects}")
