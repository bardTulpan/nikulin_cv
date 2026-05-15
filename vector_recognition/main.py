import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label as sk_label, regionprops
from skimage.io import imread
from pathlib import Path

output_dir = Path(__file__).parent / 'out'
output_dir.mkdir(exist_ok=True)

def compute_holes(region):
    h, w = region.image.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    labeled = sk_label(inverted)
    return labeled.max() - 1

def count_straight_lines(region):
    h, w = region.image.shape
    img = region.image
    vertical = (np.sum(img, axis=0) / h == 1).sum()
    horizontal = (np.sum(img, axis=1) / w == 1).sum()
    return vertical, horizontal

def extract_features(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    perimeter_norm = region.perimeter / region.image.size
    holes = compute_holes(region)
    vlines, hlines = count_straight_lines(region)
    vlines /= region.image.shape[1]
    hlines /= region.image.shape[0]
    ecc = region.eccentricity
    aspect_ratio = region.image.shape[0] / region.image.shape[1]
    area_norm = region.area / region.image.size
    return np.array([area_norm, cy, cx, perimeter_norm, holes, vlines, hlines, ecc, aspect_ratio])

def classify_region(region, templates_dict):
    feats = extract_features(region)
    best_match = ''
    min_distance = float('inf')
    for char, template_feats in templates_dict.items():
        distance = np.sqrt(((template_feats - feats) ** 2).sum())
        if distance < min_distance:
            best_match = char
            min_distance = distance
    return best_match

template_img = imread('alphabet-small.png')[:, :, :3]
template_binary = template_img.sum(axis=2) != 765
template_labels = sk_label(template_binary)
template_props = regionprops(template_labels)
template_props = sorted(template_props, key=lambda r: r.bbox[1])

symbols = ['8', '0', 'A', 'B', '1', 'W', 'X', '*', '/', '-']
templates = {s: extract_features(r) for s, r in zip(symbols, template_props)}

image = imread('alphabet.png')[:, :, :3]
binary_image = image.mean(axis=2) > 0
labeled_img = sk_label(binary_image)
print(f"Detected regions: {labeled_img.max()}")

props = regionprops(labeled_img)
recognized_counts = dict()

plt.figure(figsize=(6, 8))

for region in props:
    char = classify_region(region, templates)
    recognized_counts[char] = recognized_counts.get(char, 0) + 1

    plt.cla()
    plt.title(f'Character: "{char}"')
    plt.imshow(region.image, cmap='gray')
    plt.savefig(output_dir / f'region_{region.label}.png')

print("Recognition results:", recognized_counts)

plt.imshow(binary_image, cmap='gray')
plt.show()
