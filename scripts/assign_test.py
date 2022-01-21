import random

import numpy as np
from cv2 import cv2

from centrack.annotation import Centre, Contour
from centrack.score import assign

width = 1024
height = 1024

image = np.zeros((height, width), dtype='uint8')
inset = 50
radius = 5
offset = radius


def main():
    random.seed(1993)
    # Generate image with synthetic nuclei
    nuclei_centres = np.random.randint(inset, width - inset, (30, 2))
    for r, c in nuclei_centres:
        cv2.circle(image, (r, c), radius, 255, thickness=-1)
    cv2.imwrite('../out/assign_test.png', image)

    # Load the synthetic image
    data = cv2.imread('../out/assign_test.png', cv2.IMREAD_GRAYSCALE)
    data_bgr = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

    # Add foci to the nuclei
    _foci = np.random.randint(inset, width - inset, (30, 2))
    foci_positions = [
        (r + offset + random.randint(-3 * offset, 3 * offset), c + offset + random.randint(-3 * offset, 3 * offset)) for
        r, c in
        nuclei_centres]
    foci_detected = [Centre(f) for f in foci_positions]

    for focus in foci_positions:
        r, c = focus
        cv2.circle(data_bgr, (r, c), radius // 3, (127, 127, 127), thickness=-1)
    for focus in foci_detected:
        r, c = focus.centre
        cv2.circle(data_bgr, (r, c), radius // 3, (255, 255, 255), thickness=-1)

    # Extract the nuclei into contours
    nuclei_contours, hierarchy = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuclei_detected = [Contour(cnt) for cnt in nuclei_contours]

    # for nucleus in nuclei_detected:
    #     nucleus.draw(data_bgr, annotation=False)

    for focus in foci_detected:
        pt = focus.centre
        distances_to_nuclei = [cv2.pointPolygonTest(nucleus.contour, pt, measureDist=True) for nucleus in nuclei_detected]
        print(min(distances_to_nuclei))

    # Assign the foci to their nearest nuclei
    res = assign(foci_list=foci_detected, nuclei_list=nuclei_detected)

    # Draw the assignment
    for focus, nucleus in res:
        nucleus.draw(data_bgr, annotation=False)
        start = focus.centre
        end = nucleus.centre.centre#[::-1]
        print(start, end)
        cv2.line(data_bgr, start, end, (0, 255, 0), 3, lineType=cv2.FILLED)
        focus.draw(data_bgr)

    cv2.imwrite('../out/assign_test_res.png', data_bgr)


if __name__ == '__main__':
    main()
