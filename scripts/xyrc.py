import numpy as np
from cv2 import cv2
from labelbox.data.annotation_types import Point


def main():
    r, c = 5, 200
    canvas = np.zeros((256, 256), dtype='uint8')
    cv2.circle(canvas, (r,c), 20, 255, cv2.FILLED)  # RC

    cv2.putText(canvas, f'r={r}, c={c}',
                org=(r, c),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.8, thickness=2, color=128)

    cv2.drawContours(canvas, np.matrix([[r, c],
                                        [0, 70],
                                        [100, 250]
                                        ]), -1, 200, thickness=10)

    cv2.drawMarker(canvas, (r, c), 0, markerType=cv2.MARKER_STAR,
                   markerSize=20)

    point = Point(x=r, y=c)
    point.draw(height=10, width=10, canvas=canvas)



    cv2.imwrite('../out/coordinates.png', canvas)


if __name__ == '__main__':
    main()
