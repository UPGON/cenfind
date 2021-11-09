import logging

import cv2
import numpy as np
import tifffile as tf

from centrack.metrics import generate_synthetic_data, generate_predictions, compute_metrics

logging.basicConfig(level=logging.INFO)

CYAN_RGB = (0, 255, 255)
RED_RGB = (255, 0, 0)
WHITE_RGB = (255, 255, 255)


def main():
    height = 128
    size = 10
    positions = generate_synthetic_data(height=height, size=size)
    predictions = generate_predictions(height=height, foci=positions,
                                       fp_rate=.1, fn_rate=.05)

    results = compute_metrics(positions, predictions, offset_max=2)

    fps = results['fp']
    fns = results['fn']
    tps = results['tp']

    image = np.zeros((height, height, 3), dtype=np.uint8)

    # Generate the ground truth image
    for focus in positions:
        r, c = focus
        cv2.circle(image, radius=2, center=(r, c), color=WHITE_RGB, thickness=-1)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    annotation = np.zeros_like(image)

    for idx in fps:
        r, c = predictions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=RED_RGB, thickness=2,
                       markerSize=int(10 / np.sqrt(2)), markerType=cv2.MARKER_TILTED_CROSS)

    for idx in tps:
        r, c = positions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=CYAN_RGB, thickness=1,
                       markerSize=10, markerType=cv2.MARKER_SQUARE)

    for idx in fns:
        r, c = positions[idx]
        cv2.drawMarker(annotation, position=(r, c), color=RED_RGB, thickness=2,
                       markerSize=10, markerType=cv2.MARKER_CROSS)

    # Write the image
    result = cv2.addWeighted(image, .8, annotation, .5, 0)
    tf.imwrite('out/synthetic.png', result)

    metrics = {
        'fp': len(fps),
        'fn': len(fns),
        'tp': len(tps)
    }

    logging.info(metrics)

    logging.info('precision : %.3f', metrics['tp'] / (metrics['tp'] + metrics['fp']))
    logging.info('recall : %.3f', metrics['tp'] / (metrics['tp'] + metrics['fn']))


if __name__ == '__main__':
    main()
