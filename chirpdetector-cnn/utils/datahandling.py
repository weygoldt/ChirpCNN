import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from scipy.interpolate import interp1d

from .logger import make_logger

logger = make_logger(__name__)


def get_closest_indices(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # check if values are sorted
    sorter = None
    if np.any(np.diff(array) < 0):
        # if not, sort them
        sorter = np.argsort(array)
        array = array[sorter]

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # resort to original order
    if sorter is not None:
        idxs = sorter[np.argsort(idxs)]

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return idxs


def interpolate(data, stride):
    """
    Interpolate the tracked frequencies to a regular sampling
    """
    track_freqs = []
    track_idents = []
    track_indices = []
    new_times = np.arange(data.track_times[0], data.track_times[-1], stride)
    index_helper = np.arange(len(new_times))
    ids = np.unique(data.track_idents[~np.isnan(data.track_idents)])
    for track_id in ids:
        start_time = data.track_times[
            data.track_indices[data.track_idents == track_id][0]
        ]
        stop_time = data.track_times[
            data.track_indices[data.track_idents == track_id][-1]
        ]
        times_full = new_times[
            (new_times >= start_time) & (new_times <= stop_time)
        ]
        times_sampled = data.track_times[
            data.track_indices[data.track_idents == track_id]
        ]
        times_sampled = np.append(times_sampled, times_sampled[-1])
        freqs_sampled = data.track_freqs[data.track_idents == track_id]
        freqs_sampled = np.append(freqs_sampled, freqs_sampled[-1])

        # remove duplocates on the time array
        times_sampled, index = np.unique(times_sampled, return_index=True)

        # remove the same value on the freq array
        freqs_sampled = freqs_sampled[index]

        f = interp1d(times_sampled, freqs_sampled, kind="cubic")
        freqs_interp = f(times_full)

        index_interp = index_helper[
            (new_times >= start_time) & (new_times <= stop_time)
        ]
        ident_interp = np.ones(len(freqs_interp)) * track_id

        track_idents.append(ident_interp)
        track_indices.append(index_interp)
        track_freqs.append(freqs_interp)

    track_idents = np.concatenate(track_idents)
    track_freqs = np.concatenate(track_freqs)
    track_indices = np.concatenate(track_indices)

    data.track_idents = track_idents
    data.track_indices = track_indices
    data.track_freqs = track_freqs
    data.track_times = new_times
    return data


def resize_tensor_image(image, length):
    """
    Resize an image tensor to the specified length.

    Parameters
    ----------
    image : torch.Tensor, required
        The input image tensor.
    length : int, required
        The desired length for the resized image.

    Returns
    -------
    resized_image : torch.Tensor
        The resized image tensor.
    """
    # Get the input tensor dimensions
    image_dims = len(image.size())

    if image_dims == 2:
        # If input is of shape (height, width), add a batch and channel dimension
        image = image.unsqueeze(0).unsqueeze(0)
    elif image_dims == 3:
        # If input is of shape (channels, height, width), add a batch dimension
        image = image.unsqueeze(0)

    # Perform resizing using torch.nn.functional.interpolate
    try:
        resized_image = F.interpolate(image, size=(length, length), mode="area")
    except:
        embed()

    return resized_image


def find_on_time(array, target, limit=True):
    """Takes a time array and a target (e.g. timestamp) and returns an index
    for a value of the array that matches the target most closely.

    The time array must (a) contain unique values and (b) must be sorted from
    smallest to largest. If limit is True, the function checks for each target,
    if the difference between the target and the closest time on the time array
    is not larger than half of the distance between two time points at that
    place. When the distance exceed half the delta t, an error is returned.
    This also means that the time array must not nessecarily have a constant
    delta t.

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
        return logger.error("Target is outside of array limits.")

    def logwarning():
        return logger.warning(
            "Target is outside of array limits but you allowed this!"
        )

    # find the closest value
    idx = find_closest(array, target)

    # compute dt at this point
    found = array[idx]
    dt_target = target - found

    if target <= array[0]:
        dt_sampled = array[idx + 1] - array[idx]

        if abs(array[idx] - target) > dt_sampled / 2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()

    if target > array[0] and target < array[-1]:
        if dt_target >= 0:
            dt_sampled = array[idx + 1] - array[idx]
        else:
            dt_sampled = array[idx] - array[idx - 1]

        if abs(array[idx] - target) > dt_sampled / 2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()

    if target >= array[-1]:
        dt_sampled = array[idx] - array[idx - 1]

        if abs(array[idx] - target) > dt_sampled / 2:
            if limit:
                idx = np.nan
                logerror()
            else:
                logwarning()
    return idx


def merge_duplicates(timestamps, threshold):
    """
    Compute the mean of groups of timestamps that are closer to the previous
    or consecutive timestamp than the threshold, and return all timestamps that
    are further apart from the previous or consecutive timestamp than the
    threshold in a single list.

    Parameters
    ----------
    timestamps : List[float]
        A list of sorted timestamps
    threshold : float, optional
        The threshold to group the timestamps by, default is 0.5

    Returns
    -------
    List[float]
        A list containing a list of timestamps that are further apart than
        the threshold and a list of means of the groups of timestamps that
        are closer to the previous or consecutive timestamp than the threshold.
    """
    # Initialize an empty list to store the groups of timestamps that are
    # closer to the previous or consecutive timestamp than the threshold
    groups = []
    # Initialize an empty list to store timestamps that are further apart
    # than the threshold
    outliers = []

    # initialize the previous timestamp with the first timestamp
    prev_ts = timestamps[0]

    # initialize the first group with the first timestamp
    group = [prev_ts]

    for i in range(1, len(timestamps)):
        # check the difference between current timestamp and previous
        # timestamp is less than the threshold
        if timestamps[i] - prev_ts < threshold:
            # add the current timestamp to the current group
            group.append(timestamps[i])
        else:
            # if the difference is greater than the threshold
            # append the current group to the groups list
            groups.append(group)

            # if the group has only one timestamp, add it to outliers
            if len(group) == 1:
                outliers.append(group[0])

            # start a new group with the current timestamp
            group = [timestamps[i]]

        # update the previous timestamp for the next iteration
        prev_ts = timestamps[i]

    # after iterating through all the timestamps, add the last group to the
    # groups list
    groups.append(group)

    # if the last group has only one timestamp, add it to outliers
    if len(group) == 1:
        outliers.append(group[0])

    # get the mean of each group and only include the ones that have more
    # than 1 timestamp
    means = [np.mean(group) for group in groups if len(group) > 1]

    # return the outliers and means in a single list
    return np.sort(outliers + means)


def cluster_peaks(arr, thresh=0.5):
    """Clusters peaks of probabilitis between 0 and 1.
    Returns a list of lists where each list contains the indices of the
    all values belonging to a peak i.e. a cluster.

    Parameters
    ----------
    arr : np.ndarray
        Array of probabilities between 0 and 1.
    thresh : float, optional
        All values below are not peaks, by default 0.5

    Returns
    -------
    np.array(np.array(int))
        Each subarray contains the indices of the values belonging to a peak.
    """
    clusters = []
    cluster = []
    for i, val in enumerate(arr):
        # do nothing or append prev cluste if val is below threshold
        # then clear the current cluster
        if val <= thresh:
            if len(cluster) > 0:
                clusters.append(cluster)
                cluster = []
            continue

        # if larger than thresh
        # if first value in array, append to cluster
        # since there is no previous value to compare to
        if i == 0:
            cluster.append(i)

        # if this is the last value then there is no future value
        # to compare to so append to cluster
        elif i == len(arr) - 1:
            cluster.append(i)
            clusters.append(cluster)

        # if were at a trough then the diff between the current value and
        # the previous value will be negative and the diff between the
        # future value and the current value will be positive
        elif val - arr[i - 1] < 0 and arr[i + 1] - val > 0:
            cluster.append(i)
            clusters.append(cluster)
            cluster = []
            cluster.append(i)

        # if this is not the first value or the last value or a trough
        # then append to cluster
        else:
            cluster.append(i)

    return clusters


def norm_tensor(tensor):
    return (tensor - torch.min(tensor)) / (
        torch.max(tensor) - torch.min(tensor)
    )
