import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import label as sk_label
from scipy.optimize import linear_sum_assignment

data_folder = Path('out')
npy_files = sorted(data_folder.glob('*.npy'))


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_center(img, val):
    positions = np.argwhere(img == val)
    return positions.mean(axis=0)


def extract_centers(file_path):
    arr = np.load(file_path)
    labeled_img = sk_label(arr)
    centers_list = []
    for lbl in sorted(np.unique(labeled_img))[1:]:
        centers_list.append(compute_center(labeled_img, lbl))
    return np.array(centers_list)


def build_distance_matrix(prev, current):
    n = len(prev)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = euclidean_distance(prev[i], current[j])
    return dist_mat


num_objects = 3
trajectories = [[] for _ in range(num_objects)]
prev_positions = None
velocities = np.zeros((num_objects, 2))

for t, file_path in enumerate(npy_files):
    current_positions = extract_centers(file_path)

    if t == 0:
        prev_positions = current_positions
        for k in range(num_objects):
            trajectories[k].append(prev_positions[k])
        continue

    predicted_positions = prev_positions + velocities
    dist_matrix = build_distance_matrix(predicted_positions, current_positions)
    row_idx, col_idx = linear_sum_assignment(dist_matrix)

    new_positions_ordered = np.zeros_like(prev_positions)

    for old_idx, new_idx in zip(row_idx, col_idx):
        detected_pos = current_positions[new_idx]
        movement = euclidean_distance(detected_pos, prev_positions[old_idx])

        if movement > 35 and t > 5:
            updated_pos = prev_positions[old_idx] + velocities[old_idx]
        else:
            updated_pos = detected_pos

        velocities[old_idx] = updated_pos - prev_positions[old_idx]
        trajectories[old_idx].append(updated_pos)
        new_positions_ordered[old_idx] = updated_pos

    prev_positions = new_positions_ordered

plt.figure(figsize=(10, 10))
colors = ['red', 'blue', 'green']

for idx, traj in enumerate(trajectories):
    traj_arr = np.array(traj)
    plt.plot(traj_arr[:, 1], traj_arr[:, 0], color=colors[idx],
             marker='o', markersize=2, linewidth=1, alpha=0.7,
             label=f'Object {idx + 1}')

plt.gca().invert_yaxis()
plt.legend()
plt.show()
