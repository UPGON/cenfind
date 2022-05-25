from src.centrack.misc.train_stardist import load_set
from stardist import gputools_available
from stardist.models import Config2D
from pathlib import Path
import tifffile as tf
from csbdeep.utils import normalize
from tqdm import tqdm
from src.centrack.commands.score import get_model_stardist
from matplotlib import pyplot as plt
from stardist import random_label_cmap

lbl_cmap = random_label_cmap()


def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap='gray', clim=(0, 1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


def main():
    n_channel = 1

    path_dataset = Path('/data1/centrioles/rpe')
    path_images = path_dataset / 'projections'
    path_masks_cells = path_dataset / 'annotations/cells'
    test_set = load_set(path_dataset / 'test.txt')

    images = [str(path_images / f"{fov}_max_C2.tif") for fov in test_set]
    masks = [str(path_masks_cells / f"{fov}_max_C2.tif") for fov in test_set]

    images = list(map(tf.imread, images))
    masks = list(map(tf.imread, masks))

    images = [normalize(x, 1, 99.8) for x in tqdm(images)]

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on sub-sampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
    )
    model = get_model_stardist('/home/buergy/projects/centrack/models/stardist_centrin')

    plot_img_label(images[0], masks[0], lbl_title="label GT")
    dst = Path('/home/buergy/projects/centrack/publication/data/groundtruth.png')
    plt.gcf()
    plt.savefig(dst)

    Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
                  for x in tqdm(images)]

    plot_img_label(images[0], Y_val_pred[0], lbl_title="label Pred")
    dst = Path('/home/buergy/projects/centrack/publication/data/prediction.png')
    plt.gcf()
    plt.savefig(dst)


if __name__ == '__main__':
    main()
