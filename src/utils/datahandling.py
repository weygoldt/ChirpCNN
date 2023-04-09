import numpy as np
import cv2

from .logger import make_logger 

logger = make_logger(__name__)


def resize_image(image, length):
    image = cv2.resize(image, (length, length), interpolation = cv2.INTER_AREA)
    return image


def find_on_time(array, target, limit=True):
    """Takes a time array and a target (e.g. timestamp) and returns an index for a value of the array that matches the target most closely.

    The time array must (a) contain unique values and (b) must be sorted from smallest to largest. If limit is True, the function checks for each target, if the difference between the target and the closest time on the time array is not larger than half of the distance between two time points at that place. When the distance exceed half the delta t, an error is returned. This also means that the time array must not nessecarily have a constant delta t.

    Parameters
    ----------
    array : array, required
        The array to search in, must be sorted.
    target : float, required
        The number that needs to be found in the array.
    limit : bool, default True
        To limit or not to limit the difference between target and array value.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """

    def find_closest(array, target):
        idx = array.searchsorted(target)
        idx = np.clip(idx, 1, len(array) - 1)
        left = array[idx - 1]
        right = array[idx]
        idx -= target - left < right - target

        return idx

    def logerror(): 
        return logger.error('Target is outside of array limits.')

    def logwarning(): 
        return logger.warning('Target is outside of array limits but you allowed this!')

    # find the closest value
    idx = find_closest(array, target)

    # compute dt at this point
    found = array[idx]
    dt_target = target - found

    if target <= array[0]:
        dt_sampled = array[idx+1]-array[idx]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()

    if target > array[0] and target < array[-1]:
        if dt_target >= 0:
            dt_sampled = array[idx+1]-array[idx]
        else:
            dt_sampled = array[idx]-array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()

    if target >= array[-1]:
        dt_sampled = array[idx] - array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()
    return idx

