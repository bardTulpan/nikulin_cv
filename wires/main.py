import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as sk_label
from skimage.morphology import opening as sk_opening, dilation, closing, erosion

img = np.load("wires5.npy")

kernel = np.ones((3, 1))

labeled_img = sk_label(img)

opened_img = sk_opening(img, footprint=kernel)

num_wires = labeled_img.max()
print(f"Total wires: {num_wires}")

for wire_id in range(1, num_wires + 1):
    single_wire = (labeled_img == wire_id)
    cleaned_wire = sk_label(sk_opening(single_wire, footprint=kernel))
    parts_count = cleaned_wire.max()
    print(f"Wire {wire_id} has {parts_count} parts")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(opened_img, cmap='gray')
axes[1].set_title("After Opening")
plt.show()
