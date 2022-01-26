import json
import os
import contextlib
import logging

from cv2 import cv2

from centrack.data import Channel, Field, DataSet
from centrack.score import assign
from centrack.utils import (parse_args,
                            parse_ds_name,
                            extract_nuclei,
                            extract_centriole,
                            prepare_background,
                            draw_annotation)

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def cli():
    logging.info('Starting Centrack...')
    dataset_regex_hatzopoulos = r'^([a-zA-Z1-9]+)_(?:([a-zA-Z1-9]+)_)?([A-Z1-9+]+)_(\d)$'
    args = parse_args()

    dataset_path = args.dataset
    logging.info('Working at %s', dataset_path)

    condition = parse_ds_name(dataset_path, dataset_regex_hatzopoulos)
    dataset = DataSet(dataset_path, condition)

    if not args.out:
        projections_path = dataset.projections
    else:
        projections_path = args.out

    marker = args.marker
    if marker not in condition.markers:
        raise ValueError(f'Marker {marker} not in dataset ({condition.markers}).')

    fields = tuple(f for f in dataset.projections.glob('*.tif') if not f.name.startswith('.'))
    logging.info('%s files were found', len(fields))
    if args.test:
        fields = fields[0]

    for path in fields:
        logging.info('Loading %s', path.name)
        field = Field(path, dataset)
        data = field.load()

        logging.info('Detecting the objects...')
        foci = Channel(data)[marker].to_numpy()
        nuclei = Channel(data)['DAPI'].to_numpy()
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            foci_detected = extract_centriole(foci)
            nuclei_detected = extract_nuclei(nuclei)
        logging.info('Done (%s foci, %s nuclei)', len(foci_detected), len(nuclei_detected))

        logging.info('Assigning foci to nuclei.')
        assigned = assign(foci_list=foci_detected, nuclei_list=nuclei_detected)

        if args.out:
            logging.info('Creating annotation image.')
            background = prepare_background(nuclei, foci)
            annotation = draw_annotation(background, assigned, foci_detected, nuclei_detected)
            args.out.mkdir(exist_ok=True)
            destination_path = projections_path / f'{path.stem}_annot.png'
            successful = cv2.imwrite(str(destination_path), annotation)

            if successful:
                logging.info('Saved at %s', destination_path)

            # with open(args.out / 'dump.json', 'w') as fh:
            #     json.dump(foci_detected, fh)


if __name__ == '__main__':
    cli()
