import cv2
import numpy as np
import os
from tqdm import tqdm

from .image import imreconstruct

__all__ = ["write_video", "intersect_track", "optical_flow"]


def write_video(fgen, path, fps, tot_frame=None):
    """
    Write the video to given path with given fps.

    This method automaticall detects the color and size of the video
    and save it as file.

    Parameters
    ==========

    fgen : generator of frames

    path : str

    fps : int

    tot_frame : int, optional
        If passed, progress bar is shown on the console.

    """

    codec_dict = {"mp4": "mp4v", "avi": "DIVX"}
    name, ext = os.path.splitext(path)
    name = os.path.split(name)[-1]
    ext = ext.lstrip(".")
    fourcc = cv2.VideoWriter_fourcc(*codec_dict[ext])

    firstframe = next(fgen)

    dim = len(firstframe.shape)
    if dim == 3:
        color = True
    elif dim == 2:
        color = False
    else:
        raise ValueError("Dimension of the frame is %s" % dim)

    size = tuple(reversed(firstframe.shape[:2]))

    out = cv2.VideoWriter(path, fourcc, fps, size, isColor=color)

    out.write(firstframe)
    if tot_frame is None:
        for f in fgen:
            out.write(f)
    else:
        tot_frame = int(tot_frame)
        for i in tqdm(range(tot_frame), desc=name):
            if i == 0:
                # first frame is already written
                continue
            try:
                f = next(fgen)
            except StopIteration:
                break
            out.write(f)
    out.release()


def intersect_track(labels, frame, orig_moments=None):
    """
    Track the particles using maximum intersections. For each connected
    component in *labels*, find the largest intersecting region in
    *frame*, reconstruct the image using that regions and generate another
    labels from the result.

    Parameters
    ==========

    labels : np.ndarray
        Labelled particles. This can be retrieved from
        ``cv2.connectedComponents``.

    frame : iterable of np.ndarray
        Binarized frame. Particles must be white and background must be
        black.

    orig_moments : dict, optional
        Moment of particles in *labels*. Key is label number and value is
        coordinate of moment in tuple. Pass this parameter to save computing
        time.

    Returns
    =======

    result_label : np.ndarray
        Labelled particles in *frame*. Each particle has same label to the
        corresponding particle in *labels*.

    moment_dict : dict
        Change of image moment of ith particle. Key is label number and
        value is list of two tuples - original moment and new moment

    Notes
    =====

    *labels* must not have any noise, but minor noise in *frame* is tolerable.

    This function works for the case where particle displacement is small so
    that particle in ith and (i+1)th frame overlaps.
    """
    labels = labels.astype(frame.dtype)
    _, labels_bin = cv2.threshold(labels, 0, 255, cv2.THRESH_BINARY)

    result_label = np.zeros(frame.shape)
    ptcl_num = np.max(labels)

    # intersecting regions of every particles
    intrsct = cv2.bitwise_and(labels_bin, frame)

    moment_dict = {}
    for i in range(1, ptcl_num+1):
        # 1. mask intersecting regions of other particles
        mask = (labels == i)
        ith_intrsct = mask*intrsct
        # skip if no intersecting area is found (particle is lost)
        if not np.any(ith_intrsct):
            continue

        # 1+. find momentum of original particle location
        if orig_moments is None:
            orig_i = mask*labels_bin
            orig_i_cont, _ = cv2.findContours(orig_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orig_i_cont = sorted(orig_i_cont, key=cv2.contourArea, reverse=True)
            orig_i_cont = orig_i_cont[0]
            orig_i_mmt = cv2.moments(orig_i_cont)
            if orig_i_mmt['m00'] == 0:
                # zero moment : cannot track particle
                continue
            orig_i_x = int(orig_i_mmt['m10']/orig_i_mmt['m00'])
            orig_i_y = int(orig_i_mmt['m01']/orig_i_mmt['m00'])
        else:
            orig_i_x = orig_moments[i][0]
            orig_i_y = orig_moments[i][1]

        # 2. select the largest intersecting region of ith particle
        i_regnum, i_labels, i_stats, _ = cv2.connectedComponentsWithStats(ith_intrsct)
        max_label = max(range(1, i_regnum), key=lambda x: i_stats[x, cv2.CC_STAT_AREA])
        intrsct_mask = (i_labels == max_label)
        ith_intrsct = intrsct_mask*ith_intrsct

        # 3. reconstruct ith particle in frame
        ith_ptcl_in_frame = imreconstruct(ith_intrsct, frame)

        # 3+. find momentum of new particle location
        new_i_cont, _ = cv2.findContours(ith_ptcl_in_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_i_cont = sorted(new_i_cont, key=cv2.contourArea, reverse=True)
        new_i_cont = new_i_cont[0]
        new_i_mmt = cv2.moments(new_i_cont)
        if new_i_mmt['m00'] == 0:
            # zero moment : cannot track particle
            continue
        new_i_x = int(new_i_mmt['m10']/new_i_mmt['m00'])
        new_i_y = int(new_i_mmt['m01']/new_i_mmt['m00'])

        # 4. Draw ith particle in result_label with label i
        result_label[ith_ptcl_in_frame != 0] = i

        # 5. Store old momentum and new momentum
        moment_dict[i] = [(orig_i_x, orig_i_y), (new_i_x, new_i_y)]

    result_label = result_label.astype(labels.dtype)
    return result_label, moment_dict


def optical_flow(frames, feature_param, lk_param):
    """
    Perform Lucas-Kanade optical flow tracking of *frames*.
    Updates feature point at every 5 frames and runs backward-checking [1].

    Parameters
    ==========

    frames : iterable of np.ndarray
        Iterable of grayscale images

    feature_param : dict
        Parameters for ``cv2.goodFeaturesToTrack``.

    lk_param : dict
        Parameters for ``cv2.calcOpticalFlowPyrLK``.

    Yields
    ======

    trail_frame : np.ndarray
        BGR frame containing the trails of trackable features

    tracks : list
        List of past points for each trackable feature

    References
    ==========

    .. [1] https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py

    """
    track_len = 10
    detect_interval = 5
    tracks = []
    frame_idx = 0

    for frame in frames:
        prev_frame = frame

        trail_frame = np.zeros(frame.shape, dtype=np.uint8)
        trail_frame = cv2.cvtColor(trail_frame, cv2.COLOR_GRAY2BGR)

        if len(tracks) > 0:
            img0, img1 = prev_frame, frame
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_param)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_param)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            p_reshape = p1.reshape(-1, 2)
            for tr, (x, y), good_flag in zip(tracks, p_reshape, good):
                if not good_flag:
                    # Bad features are deleted
                    continue
                # New position of every feature is appended
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(trail_frame, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(trail_frame, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

        if frame_idx % detect_interval == 0:
            # Detect new trackable features
            mask = np.zeros_like(frame)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                # Do not detect existing features
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame, mask = mask, **feature_param)
            if p is not None:
                p_reshape = p.reshape(-1, 2)
                for x, y in np.float32(p_reshape):
                    tracks.append([(x, y)])

        frame_idx += 1

        yield trail_frame, tracks
