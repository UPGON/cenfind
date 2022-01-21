from cv2 import cv2

from centrack.annotation import Centre


def assign(foci_list, nuclei_list):
    """
    Assign detected centrioles to the nearest nucleus.
    1.
    :return: List[Tuple[Centre, Contour]]
    """
    if len(foci_list) == 0:
        raise ValueError('Empty foci list')
    if len(nuclei_list) == 0:
        raise ValueError('Empty nuclei list')

    assigned = []
    for c in foci_list:
        distances = [(n, cv2.pointPolygonTest(n.contour, c.centre, measureDist=True)) for n in nuclei_list]
        nucleus_nearest = max(distances, key=lambda x: x[1])
        assigned.append((c, nucleus_nearest[0]))

    return assigned


def main():
    nuclei_list = [(1, 10), (4, 5), (3, 4), (10, 2)]
    foci_list = [(2, 7), (3, 7), (20, 5), (30, 1), (2, 14), (2, 6)]

    nuclei_list = [Centre(t, idx=i, label=f'nucleus {i}', confidence=-1) for i, t in enumerate(nuclei_list)]
    foci_list = [Centre(t, idx=i, label=f'centriole {i}', confidence=-1) for i, t in enumerate(foci_list)]

    result = assign(foci_list, nuclei_list)
    print(result)


if __name__ == '__main__':
    main()
