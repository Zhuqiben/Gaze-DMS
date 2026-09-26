"""Mouth aspect ratio (MAR) for yawn detection."""

import math


def mouth_aspect_ratio(mouth):
    """Return the mouth aspect ratio from a 68 point face landmark array.

    Uses the 0-based indices of the 68 point layout: 49/56 and 51/54 are the
    vertical inner/outer lip pairs and 58/62 are the mouth corners.
    """
    vertical_1 = math.dist(mouth[49], mouth[56])
    vertical_2 = math.dist(mouth[51], mouth[54])
    horizontal = math.dist(mouth[58], mouth[62])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
