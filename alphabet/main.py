import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as sk_label, regionprops
from skimage.io import imread
from pathlib import Path

output_dir = Path(__file__).parent / "classified_symbols"
output_dir.mkdir(exist_ok=True)

def compute_holes(region):
    h, w = region.image.shape
    padded = np.zeros((h + 2, w + 2))
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    labeled = sk_label(inverted)
    return labeled.max() - 1

def count_lines(region):
    h, w = region.image.shape
    img = region.image
    vertical = (np.sum(img, axis=0) / h == 1).sum()
    horizontal = (np.sum(img, axis=1) / w == 1).sum()
    return vertical, horizontal

def symmetry_measure(region, horizontal=False):
    img = region.image.T if horizontal else region.image
    rows = img.shape[0]
    top = img[:rows // 2]
    if rows % 2 != 0:
        bottom = img[rows // 2 + 1:]
    else:
        bottom = img[rows // 2:]
    bottom = bottom[::-1]
    return (top == bottom).sum() / (top.size)

def recognize_symbol(region):
    holes = compute_holes(region)
    
    if holes == 2:  # B, 8
        v, _ = count_lines(region)
        v /= region.image.shape[1]
        return 'B' if v > 0.2 else '8'
    
    elif holes == 1:  # A, 0, D, P
        ecc = region.eccentricity
        v_sym = symmetry_measure(region)
        h_sym = symmetry_measure(region, horizontal=True)
        if v_sym > 0.989 and (ecc > 0.7 or ecc < 0.6):
            return 'D'
        if v_sym < 0.6 and h_sym > 0.7:
            return 'A'
        if h_sym > 0.8 and v_sym > 0.8:
            return 'O'
        return 'P'
    
    elif holes == 0:  # 1, W, X, *, -, /
        fill_ratio = region.image.sum() / region.image.size
        if fill_ratio > 0.95:
            return '-'
        h, w = region.image.shape
        aspect = min(h, w) / max(h, w)
        if aspect > 0.9:
            return '*'
        v_sym = symmetry_measure(region)
        h_sym = symmetry_measure(region, horizontal=True)
        if v_sym > 0.8 and h_sym > 0.8:
            return 'X'
        elif h_sym > 0.8:
            return 'W'
        v, _ = count_lines(region)
        return '1' if v > 0.5 else '/'
    
    return "?"

img = imread("symbols.png")[:, :, :3]
binary_img = img.mean(axis=2) > 0
labeled_img = sk_label(binary_img)
regions = regionprops(labeled_img)

symbol_counts = {}

plt.ion()
plt.figure(figsize=(5, 7))

for region in regions:
    char = recognize_symbol(region)
    symbol_counts[char] = symbol_counts.get(char, 0) + 1
    plt.cla()
    plt.title(f'Symbol: {char}')
    plt.savefig(output_dir / f"region_{region.label}.png")

print("Recognition summary:", symbol_counts)
