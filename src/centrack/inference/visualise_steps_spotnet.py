from spotipy.utils import normalize_fast2d
from centrack.layout.dataset import FieldOfView
from centrack.inference.score import get_model
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import numpy as np
from skimage import exposure
from centrack.visualisation.outline import prepare_background

plt.rcParams["font.family"] = "Helvetica"


def main():
    model = get_model('models/master')
    fov = FieldOfView(Path(
        '/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP63+CETN2+PCNT_1/projections/RPE1wt_CEP63+CETN2+PCNT_1_004_001_max.tif'))
    data = fov.data[2, :, :]
    dna = fov.data[0, :, :]

    percentile = .2
    width = 32
    x, y = 1214, 406
    top_left = slice(x, x + width)
    bottom_right = slice(y, y + width)

    composite = prepare_background(dna, data)
    percentiles = np.percentile(composite, (percentile, 100 - percentile))
    scaled = exposure.rescale_intensity(composite,
                                        in_range=tuple(percentiles))
    cell = composite[top_left, bottom_right]

    x = normalize_fast2d(data)
    prob_thresh = .5

    foci = model.predict(x,
                         prob_thresh=prob_thresh,
                         show_tile_progress=False)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 7.2 / 3))

    ax[0].imshow(scaled, interpolation='nearest')
    roi = patches.Rectangle((406, 1214), width, width, linewidth=.5, edgecolor='white', facecolor='none')
    ax[0].add_patch(roi)

    scalebar_fov = ScaleBar(dx=.1025,
                            units='um',
                            location='lower right',
                            box_alpha=0, color='white')
    ax[0].add_artist(scalebar_fov)
    ax[0].set_title('DNA + Centrin')

    ax[1].imshow(cell, interpolation='nearest')
    scalebar_fov = ScaleBar(dx=.1025, units='um',
                            location='lower right',
                            box_alpha=0, color='white')
    ax[1].add_artist(scalebar_fov)
    ax[1].set_title('Centrin')

    ax[2].imshow(foci[0][top_left, bottom_right], vmin=0, vmax=1,
                 cmap='cividis', interpolation='nearest')
    ax[2].set_title('Prediction')

    for a in ax:
        a.axis('off')
    fig.savefig('out/test.png', bbox_inches='tight', dpi=300)

    return 0


if __name__ == '__main__':
    main()
