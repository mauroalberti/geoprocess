# -*- coding: utf-8 -*-

from typing import Optional

import numbers

from geoprocess.stereonets.exceptions import *


def rake_to_apsg_movsense(rake: numbers.Real) -> Optional[int]:
    """
    Convert rake to movement sense according to APSG convention.

    :param rake: the lineation rake angle, in decimal degrees
    :type: numbers.Real
    :return: 1 or -1, according to the APSG convention
    :rtype: int
    """

    if not isinstance(rake, numbers.Real):
        raise RakeInputException("Input rake value must be number")

    if 0 < rake < 180:  # reverse faults according to Aki & Richards, 1980 convention
        return 1
    elif -180 < rake < 0:  # normal faults according to Aki & Richards, 1980 convention
        return -1
    elif abs(rake) == 0.0 or abs(rake) == 180.0:
        raise RakeInputException("Currently trascurrent data (rake = +/-180 or = 0.0) are not handled in plots")
    else:
        raise RakeInputException("Input rake value not acceptable")


def movsense_to_apsg_movsense(str_val: str) -> Optional[int]:
    """
    Convert text movement sense convention to APSG convention.

    :param str_val: "R" for reverse or "N" for normal
    :type str_val: str
    :return: 1 for input "R" or -1 for input "N"
    :type: int
    """

    if str_val == "R":
        return 1
    elif str_val == "N":
        return -1
    else:
        raise RakeInputException("Input rake value not acceptable")

