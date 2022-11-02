from spotipy.utils import normalize_fast2d
from cenfind.core.detectors import get_model
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
import numpy as np
from skimage import exposure
from cenfind.core.outline import to_8bit
from cenfind.core.data import Dataset, Field

plt.rcParams["font.family"] = "Helvetica"


def main():
    model = get_model('models/dev/5785b6d9-f09b-4486-af65-0a923c8ae533')

    dataset = Dataset(Path('/Users/buergy/Dropbox/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1'))
    field = Field('RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_001_002', dataset)
    data = field.projection[2, :, :]
    dna = field.projection[0, :, :]

    percentile = .2
    width = 32
    x, y = 1214, 406
    top_left = slice(x, x + width)
    bottom_right = slice(y, y + width)

    composite = np.zeros_like(data)
    composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)
    composite[:, :, 2] = to_8bit(dna)
    composite[:, :, 1] = to_8bit(data)
    percentiles = np.percentile(composite, (percentile, 100 - percentile))
    scaled = exposure.rescale_intensity(composite,
                                        in_range=tuple(percentiles),
                                        out_range='uint8')
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
