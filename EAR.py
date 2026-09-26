"""Eye aspect ratio (EAR) for blink and eye-closure detection."""

import math


def eye_aspect_ratio(eye):
    """Return the eye aspect ratio of six landmarks around one eye.

    The landmarks are expected in the order [P1, P2, P3, P4, P5, P6],
    starting from one corner of the eye, so that P2/P6 and P3/P5 are the
    vertical eyelid pairs and P1/P4 are the horizontal corners.
    """
    vertical_1 = math.dist(eye[1], eye[5])
    vertical_2 = math.dist(eye[2], eye[4])
    horizontal = math.dist(eye[0], eye[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
