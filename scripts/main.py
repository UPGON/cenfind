import contextlib
import logging
import os

from cv2 import cv2

from centrack.data import Channel, Field, DataSet
from centrack.score import assign
from centrack.utils import (parse_args,
                            condition_from_filename,
                            extract_nuclei,
                            extract_centriole,
                            prepare_background,
                            draw_annotation)

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def cli():
    logging.info('Starting Centrack...')

    filename_patterns = {
        'hatzopoulos': r'([\w\d]+)_(?:([\w\d-]+)_)?([\w\d\+]+)_(\d)',
        'garcia': r'^(?:\d{8})_([\w\d-]+)_([\w\d_-]+)_([\w\d\+]+)_((?:R\d_)?\d+)?_MMStack_Default'
    }

    args = parse_args()

    dataset_path = args.dataset
    logging.debug('Working at %s', dataset_path)

    dataset = DataSet(dataset_path)

    if not args.out:
        projections_path = dataset.projections
    else:
        projections_path = args.out
    projections_path.mkdir(exist_ok=True)

    fields = tuple(f for f in dataset.projections.glob('*.tif') if not f.name.startswith('.'))
    logging.debug('%s files were found', len(fields))

    if args.test:
        logging.warning('Test mode enabled: only one field will be processed.')
        fields = [fields[0]]

    for path in fields:
        logging.info('Loading %s', path.name)
        condition = condition_from_filename(path.name, filename_patterns['hatzopoulos'])
        field = Field(path, condition, dataset)
        data = field.load()

        marker = args.marker
        if marker not in condition.markers:
            raise ValueError(f'Marker {marker} not in dataset ({condition.markers}).')

        logging.info('Detecting the objects...')
        foci = Channel(data)[marker].to_numpy()
        nuclei = Channel(data)['DAPI'].to_numpy()

        # This skips the print calls in spotipy
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            foci_detected = extract_centriole(foci)
            nuclei_detected = extract_nuclei(nuclei)
        logging.info('%s: (%s foci, %s nuclei)', path.name, len(foci_detected), len(nuclei_detected))

        logging.debug('Assigning foci to nuclei.')
        try:
            assigned = assign(foci_list=foci_detected, nuclei_list=nuclei_detected)
        except ValueError:
            logging.warning('No foci/nuclei detected (%s)', path.name)
            continue

        if args.out:
            logging.debug('Creating annotation image.')
            background = prepare_background(nuclei, foci)
            annotation = draw_annotation(background, assigned, foci_detected, nuclei_detected)
            args.out.mkdir(exist_ok=True)
            destination_path = projections_path / f'{path.name.removesuffix(".ome.tif")}_annot.png'
            successful = cv2.imwrite(str(destination_path), annotation)

            if successful:
                logging.debug('Saved at %s', destination_path)

            # with open(args.out / 'dump.json', 'w') as fh:
            #     json.dump(foci_detected, fh)


if __name__ == '__main__':
    cli()
