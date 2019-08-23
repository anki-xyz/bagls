import numpy as np
from skimage.util import pad
# Image normalization
from keras.applications.imagenet_utils import preprocess_input
# Convert RGB data to grayscale
from skimage.color import rgb2gray
# Read images from file
from imageio import imread
# For JIT compilation of code
from numba import njit, prange

def divpad(im, multiple_of=32, cval=0):
    """Padding of images that each dimension is a multiple of a given number.
    
    Parameters
    ----------
    im : numpy.ndarray
        The input image
    multiple_of : int, optional
        the desired multiple of in each dimension, by default 32
    cval : int, optional
        the padding value, by default 0
    
    Returns
    -------
    numpy.ndarray
        The padded image
    """
    needed_padding = []
    real_padding = []
    
    for sh in im.shape:
        if sh > 3 and sh % multiple_of:
            needed_padding.append(multiple_of - sh % multiple_of)
        else:
            needed_padding.append(0)

    real_padding.append([needed_padding[0]//2, 
                        needed_padding[0]//2+needed_padding[0] % 2])

    real_padding.append([needed_padding[1]//2, 
                        needed_padding[1]//2+needed_padding[1] % 2])
    
    return pad(im, real_padding, 'constant', constant_values=cval)

def load(im_fn, gt_fn):
    """
    Load images from file and apply preprocessing.
    Images are load from file, converted from RGB to grayscale,
    then padded that each dimension is divisable by 32,
    and in case of the input image normalized to [-1, 1].

    Parameters
    ----------
    im_fn : str
        Path to image file
    gt_fn : str
        Path to segmentation (i.e. ground truth) file

    Returns
    -------
    tuple
        of the image and the segmentation

    """
    image = divpad(rgb2gray(imread(im_fn)) * 255)
    mask = divpad(rgb2gray(imread(gt_fn)) / 255)

    image = preprocess_input(image, mode='tf')

    return image, mask.round()

@njit(parallel=True)
def IoU(ground_truth, prediction):
    """
    Intersection over union

    Parameters
    ----------
    ground_truth : ndarray
        Array with shape time x X x Y
    b : ndarray
        Array with shape time x X x Y

    Returns
    -------
    ndarray
        with shape time

    """
    iou = np.zeros((len(ground_truth), 2), dtype=np.float32)

    # Iterate over images
    for i in prange(len(ground_truth)):
        target_shape = ground_truth[i].shape[0:2]
        gt = ground_truth[i].reshape(target_shape)
        pred = prediction[i].reshape(target_shape)

        # Iterate over X and Y
        for j in range(gt.shape[0]):
            for k in range(gt.shape[1]):
                tmp_gt = gt[j, k]
                tmp_pred = pred[j, k]

                # Intersection and Union
                if tmp_gt == np.bool_(True) and tmp_pred == np.bool_(True):
                    iou[i, 0] += 1
                    iou[i, 1] += 1
                # Union
                elif tmp_gt != tmp_pred:
                    iou[i, 1] += 1
 
        # if no instances of class 1 are present and shouldn't be
        if iou[i,0] == 0 and iou[i,1] == 0:   
            iou[i,0] = 1
            iou[i,1] = 1

    # Intersection over the Union
    return iou[:, 0] / iou[:, 1]