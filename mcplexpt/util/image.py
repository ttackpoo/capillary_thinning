import cv2
import numpy as np

__all__ = ["measure_pixels", "imreconstruct"]


def measure_pixels(frame):
    """
    Open a window that allows the user to draw a rectangle on the frame
    (without saving it). Coordinates of two corners of a rectangle and
    the number of pixels between them are printed.

    This allows to measure the length between two points in pixel.
    Window closes by pressing 'q'.

    Parameters
    ==========

    frame : numpy.ndarray

    """

    if len(frame.shape) == 2:
        # frame is gray. convert to color.
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    param = {"p1": (), "p2": (), "clicked": False}

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["clicked"] = True
            param["p1"] = (x, y)
            param["p2"] = ()
        elif event == cv2.EVENT_MOUSEMOVE:
            if param["clicked"] is True:
                param["p2"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            param["clicked"] = False
            p1, p2 = param["p1"], param["p2"]
            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            print("Point : %s, %s" % (p1, p2))
            print("# of pixels : %s" % d)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw, param=param)

    while True:
        temp = frame.copy()
        p1, p2 = param["p1"], param["p2"]
        if p1 and p2:
            cv2.rectangle(temp, p1, p2, (0, 255, 0), 1)
        cv2.imshow('image', temp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


def imreconstruct(marker, mask):
    """
    Iteratively expand the markers white keeping them limited by the mask
    during each iteration.

    Parameters
    ==========

    marker : np.ndarray
        Grayscale image where initial seed is white on black background.

    mask : np.ndarray
        Grayscale mask where the valid area is white on black background.

    Returns
    =======

    expanded : np.ndarray
        A copy of the last expansion.

    Notes
    =====

    Written By Semnodime.
    https://gist.github.com/Semnodime/ddf1e63d4405084f886204e73ecfabcd
    """
    kernel = np.ones(shape=(3,3), dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded
