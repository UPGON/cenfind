import numpy as np
import cv2

if __name__ == '__main__':
    shape = (200, 300) # H, W => row col
    r, c = 20, 50 # x, y

    mask = np.zeros(shape, dtype='uint8')
    cv2.drawMarker(mask, (r, c), (255, 255, 255), markerType=cv2.MARKER_SQUARE,
                   markerSize=8)
    cv2.putText(mask, f'Text at x={r} y={c}',
                        org=(r, c), # x y from top right
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.4, thickness=1, color=(255, 255, 255))

    cv2.imwrite('../../../out/coordinates.png', mask)
