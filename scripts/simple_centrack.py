import logging
import tifffile as tf


def load_ome(path):
    """
    Loads a raw OME TIFF file in memory using tifffile.
    :return: a header dict, and a Numpy array
    """

    with tf.TiffFile(path) as file:
        shape = file.series[0].shape

        order = file.series[0].axes
        dimensions_found = set(order.lower())
        dimensions_expected = set('czyx')

        if dimensions_found != dimensions_expected:
            raise ValueError(f"Dimension mismatch: found: {dimensions_found} vs expected: {dimensions_expected}")
        if order == 'ZCYX':
            z, c, y, x = shape
        elif order == 'CZYX':
            c, z, y, x = shape
        else:
            raise ValueError(f'Order is not understood {order}')

        micromanager_metadata = file.micromanager_metadata

        if micromanager_metadata:
            pixel_size_um = micromanager_metadata['Summary']['PixelSize_um']
            logging.warning('No pixel size could be found')
        else:
            pixel_size_um = None

        if pixel_size_um:
            pixel_size_cm = pixel_size_um / 1e4
        else:
            pixel_size_cm = None

        data = file.asarray()

        logging.info('Order: %s Shape: %s', order, shape)
        result = data.reshape((c, z, y, x))

    return pixel_size_cm, result


def project(data, ):
    """
    Projects a field across the Z-axis.
    :return: a H x W numpy array
    """
    result = data.max(axis=1)
    return result


def write_projection(dst, data, pixel_size=None):
    """
    Writes the projection to the disk.
    """
    if pixel_size:
        res = (1 / pixel_size, 1 / pixel_size, 'CENTIMETER')
    else:
        res = None
    tf.imwrite(dst, data, photometric='minisblack', resolution=res)


def detect_foci():
    """
    Runs the detection algorithm
    :return: List of position of the detected foci
    """
    return result


def segment_nuclei():
    """
    Runs the segmentation algorithm for the cell contours.
    :return: A mask of the images with labelled pixels
    """
    return result


def assign(foci, nuclei):
    """
    Assigns the foci to the nuclei using the specified method.
    :return: a mapping of foci to their cell they belong to.
    """
    return result


def main():
    pixel_size, field = load_ome('/Volumes/work/epfl/datasets/RPE1wt_CEP152+GTU88+PCNT_1/raw/RPE1wt_CEP152+GTU88+PCNT_1_MMStack_1-Pos_000_001.ome.tif')
    projection_foci = project(field, channel=1)
    projection_nuclei = project(field, channel=0)

    foci = detect_foci(projection_foci)
    nuclei = segment_nuclei(projection_nuclei)

    res = assign(foci, nuclei)

    save_result('dst/res.json', res)


if __name__ == '__main__':
    main()
