

from math import radians, sin, cos, sqrt
from typing import Union


class Attitude:
    """
    Represent a geological attitude projected onto a vertical profile.
    """

    def __init__(
        self,
        id: int,
        s: float,
        z: float,
        slope_degr: float,
        down_sense: str,
        dist: float
):
        """
        :param id: the identifier of the observation.
        :type id: int.
        :param s: the signed horizontal distance along the profile.
        :type s: float (note: may exceed a profile range, both before, negative values, or after its end.
        :param z: the height of the attitude in the profile.
        :type z: float.
        :param slope_degr: the slope of the attitude in the profile. Unit is degrees.
        :type slope_degr: float.
        :param down_sense: downward sense, to the right or to the profile left.
        :type down_sense: str.
        :param dist: the distance between the attitude point and the point projection on the profile.
        :type: dist: float
        """

        self.id = id
        self.s = s
        self.z = z
        self.slope_degr = slope_degr
        self.down_sense = down_sense
        self.dist = dist

    def __repr__(self) -> str:
        """
        Creates the representation of a ProfileAttitude instance.

        :return: the representation of a ProfileAttitude instance.
        :rtype: str.
        """

        return"ProfileAttitude(id={}, s={}, z={}, slope_degr={}, down_sense={}, dist={})".format(
            self.id,
            self.s,
            self.z,
            self.slope_degr,
            self.down_sense,
            self.dist
        )


class Drill:

    pass


def define_plot_structural_segment(
        profile_attitude: Attitude,
        profile_length: Union[int, float],
        vertical_exaggeration: Union[int, float] = 1,
        segment_scale_factor: float = 70.0):
    """

    :param profile_attitude:
    :param profile_length:
    :param vertical_exaggeration:
    :param segment_scale_factor: the scale factor controlling the attitude segment length in the plot.
    :return:
    """

    ve = float(vertical_exaggeration)

    z0 = profile_attitude.z

    h_dist = profile_attitude.s
    slope_rad = radians(profile_attitude.slope_degr)
    intersection_downward_sense = profile_attitude.down_sense
    length = profile_length / segment_scale_factor

    s_slope = sin(float(slope_rad))
    c_slope = cos(float(slope_rad))

    if c_slope == 0.0:
        height_corr = length / ve
        structural_segment_s = [h_dist, h_dist]
        structural_segment_z = [z0 + height_corr, z0 - height_corr]
    else:
        t_slope = s_slope / c_slope
        width = length * c_slope

        length_exag = width * sqrt(1 + ve*ve * t_slope*t_slope)

        corr_width = width * length / length_exag
        corr_height = corr_width * t_slope

        structural_segment_s = [h_dist - corr_width, h_dist + corr_width]
        structural_segment_z = [z0 + corr_height, z0 - corr_height]

        if intersection_downward_sense == "left":
            structural_segment_z = [z0 - corr_height, z0 + corr_height]

    return structural_segment_s, structural_segment_z