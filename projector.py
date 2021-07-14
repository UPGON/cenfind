from pathlib import Path

from matplotlib import pyplot as plt

import tifffile as tf
import numpy as np
import cv2

from centrack import sharp_planes, image_8bit_contrast


# Create directory variables
path_root = Path('/Volumes/work/datasets/')
dataset_name = 'RPE1wt_CEP63+CETN2+PCNT_1'
path_raw = path_root / dataset_name / 'raw'

path_projections = path_root / dataset_name / 'projected'
path_projections.mkdir(exist_ok=True)

# Collect the ome.tiff files
files = sorted(tuple(file for file in path_raw.iterdir()
                     if file.name.endswith('.tif')
                     if not file.name.startswith('.')))

path_master_file = files[0]
fov = tf.imread(path_master_file)

target_dims = (4, 67, 2048, 2048)

reshaped = np.expand_dims(fov, 0)

assert reshaped.shape == target_dims, f"Not the same dimensions {fov.shape}; target={target_dims}"

markers = dataset_name.split('_')[-2].split('+')
if 'DAPI' not in markers:
    markers = ['DAPI'] + markers
markers_map = list(zip(range(len(markers)), markers))
print(f"{dataset_name} => {markers_map}")

channels_n = reshaped.shape[0]
depth_n = reshaped.shape[1]
fig, axs = plt.subplots(ncols=len(markers), figsize=(20, 5))

for i, name in markers_map:
    ax = axs[i]
    profile, projected = sharp_planes(reshaped,
                                      shape=target_dims,
                                      reference_channel=i,
                                      threshold=0)
    ax.plot(profile, 'k')
    ax.hlines(profile.mean(), 0, depth_n, colors='k')
    ax.set_title(f"{name}; {int(profile.mean())}")


channel_id = int(input("Please enter the channel"))
threshold = int(input("Please enter the threshold"))

for f, file in enumerate(files):
    print(f"Loading {file.name}")
    fov = tf.imread(path_raw / file.name)
    reshaped = np.expand_dims(fov, 0)
#     reshaped = np.moveaxis(fov, 0, 1)
#     reshaped = fov.flatten().reshape(target_dims)
#     profile, projected = sharp_planes(reshaped,
#                                       shape=target_dims,
#                                       reference_channel=channel_id,
#                                       threshold=threshold)
    projected = reshaped.max(1)
    tf.imwrite(path_projections / file.name, projected)


if __name__ == '__main__':
    main()