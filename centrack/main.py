import json
from cv2 import cv2
from centrack.data import DataSet, Field, Channel, Condition, PixelSize
from centrack.utils import parse_args, contrast
from centrack.detectors import FocusDetector


def cli():
    args = parse_args()
    row, col = [int(i) for i in args.coords]

    markers = 'DAPI+CEP152+GTU88+PCNT'.split('+')
    conditions = Condition(markers=markers,
                           genotype='RPE1wt',
                           pixel_size=PixelSize(.1025, 'um'))

    dataset_path = args.dataset
    field_name = 'RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_000_max.tif'
    ds = DataSet(dataset_path, condition=conditions)
    projection_path = dataset_path / 'projections' / field_name
    field = Field(projection_path, dataset=ds)
    data = field.load()

    foci = Channel(data)[args.marker].to_numpy()
    focus_detector = FocusDetector(foci, 'centriole')
    foci_detected = focus_detector.detect(5)

    background = cv2.cvtColor(contrast(foci), cv2.COLOR_GRAY2BGR)
    for c in foci_detected:
        c.draw(background)

    if args.out:
        args.out.mkdir(exist_ok=True)

        cv2.imwrite(str(args.out / f'{args.marker}_{row:03}_{col:03}.png'), background)
        # with open(args.out / 'dump.json', 'w') as fh:
        #     json.dump(foci_detected, fh)


if __name__ == '__main__':
    cli()
