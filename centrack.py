import json

import cv2
import tifffile as tf


def file_read(path):
    """
    Read an ome tiff file.
    :param path:
    :return: Numpy array
    """
    pass


def channel_extract(ome_tiff, channel_id):
    """
    Extract a channel and apply a projection.
    :param ome_tiff: 5D array
    :return: 2D array for the channel
    """
    pass


def centrioles_detect(image):
    """
    Detect the foci
    :param image:
    :return:
    """
    pass


def centrosomes_segment(foci_coords):
    """
    Compute the contours of neighbouring foci.
    :param foci_coords:
    :return: list of contours
    """
    pass


def cell_segment(image):
    """
    Segment the cell based on the nuclei
    :param image:
    :return:
    """
    pass


def cnt_centre(contour):
    """
    Compute the centre of a contour
    :param contour:
    :return: the coordinates of the contour
    """
    moments = cv2.moments(contour)

    c_x = int(moments['m10'] / moments['m00'])
    c_y = int(moments['m01'] / moments['m00'])

    return c_x, c_y


def main():
    path = 'Path'
    data = file_read(path)
    centrioles = channel_extract(data, 1)
    nuclei = channel_extract(data, 0)

    foci_coords = centrioles_detect(centrioles)
    cells_contours = cell_segment(nuclei)
    centrosomes_contours = centrosomes_segment(foci_coords)

    results = []

    for n, cnt_cell in enumerate(cells_contours):
        for c, cnt_cm in enumerate(centrosomes_contours):
            centre = cnt_centre(cnt_cm)
            if cv2.pointPolygonTest(cnt_cell, centre, measureDist=False) < 0:
                for f, focus in enumerate(foci_coords):
                    if cv2.pointPolygonTest(cnt_cm, focus, measureDist=False) < 0:
                        results.append((n, c, f))


if __name__ == '__main__':
    main()
