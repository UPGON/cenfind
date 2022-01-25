import numpy as np
from cv2 import cv2

from centrack.data import DataSet, Field, Channel, Condition, PixelSize
from centrack.detectors import FocusDetector, NucleiStardistDetector
from centrack.utils import parse_args, contrast
from centrack.score import assign


def cli():
    args = parse_args()
    row, col = [int(i) for i in args.coords]

    markers = 'DAPI+CEP152+GTU88+PCNT'.split('+')
    conditions = Condition(markers=markers,
                           genotype='RPE1wt',
                           pixel_size=PixelSize(.1025, 'um'))

    dataset_path = args.dataset
    field_name = f'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_{row:03}_{col:03}_max.tif'
    ds = DataSet(dataset_path, condition=conditions)
    projection_path = dataset_path / 'projections' / field_name
    field = Field(projection_path, dataset=ds)
    data = field.load()

    foci = Channel(data)[args.marker].to_numpy()
    focus_detector = FocusDetector(foci, 'Centriole')
    foci_detected = focus_detector.detect(5)

    nuclei = Channel(data)['DAPI'].to_numpy()
    nuclei_detector = NucleiStardistDetector(nuclei, 'Nucleus')
    nuclei_detected = nuclei_detector.detect()

    res = assign(foci_list=foci_detected, nuclei_list=nuclei_detected)

    background = cv2.cvtColor(contrast(nuclei), cv2.COLOR_GRAY2BGR)
    foci_bgr = np.zeros_like(background)
    foci_bgr[:, :, 2] = contrast(foci)
    background = cv2.addWeighted(background, .5, foci_bgr, 1, 1)

    for c in foci_detected:
        c.draw(background)

    for n in nuclei_detected:
        n.draw(background)

    for c, n in res:
        start = c.centre
        end = n.centre.centre
        print(start, end)
        cv2.line(background, start, end, (0, 255, 0), 3, lineType=cv2.FILLED)

    if args.out:
        args.out.mkdir(exist_ok=True)

        cv2.imwrite(str(args.out / f'{args.marker}_{row:03}_{col:03}.png'), background)
        # with open(args.out / 'dump.json', 'w') as fh:
        #     json.dump(foci_detected, fh)


if __name__ == '__main__':
    cli()
