import cv2
import tifffile as tf
import numpy as np


def sharp_planes(array, shape, reference_channel, threshold):
    """
    Compute the sharpness of the planes and max-project planes above threshold.
    
    Arguments
    ---------
    array (4D Numpy array): the stack to max-project
    threshold (integer): std threshold above which plane is considered sharp
    
    Return
    ------
    array (3D): the stack containing the max-projection of each channel
    """
    
    profile = array[reference_channel, :, :, :].std(axis=(1, 2))
    projected = array[:, profile > threshold, :, :].max(axis=1)
    print(profile.shape, projected.shape)
    
    return profile, projected


def image_8bit_contrast(image):
    return cv2.convertScaleAbs(image, alpha=255/image.max())
