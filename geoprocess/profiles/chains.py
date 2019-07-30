
from typing import List, Optional, Tuple

from array import array

from ..types.utils import check_optional_type, check_type
from .elements import ProjctAttitude


class TopographicProfile():
    """

    """

    def __init__(self, s_array: array, z_array: array):

        check_type(s_array, "Distances array", array)
        if s_array.typecode != 'd':
            raise Exception("s array must be of type double")

        num_steps = len(s_array)

        check_type(z_array, "Scalar values array", array)
        if z_array.typecode != 'd':
            raise Exception("All z arrays must be of type double")
        if len(z_array) != num_steps:
            raise Exception("All z arrays must have the same length of s array")

        self._num_steps = num_steps
        self._s = s_array
        self._z = z_array

    def s(self) -> array:
        """
        Return the s array.

        :return: the s array.
        :rtype: array.
        """

        return self._s

    def z(self) -> array:
        """
        Return the z arrays.

        :return: the z array.
        :rtype: array.
        """

        return self._z

    def num_steps(self) -> int:
        """
        Return the number of steps of the profiles.

        :return: number of steps of the profiles.
        :rtype: int.
        """

        return self._num_steps

    def profile_length(self) -> numbers.Real:
        """
        Returns the length of the profile.

        :return: length of profile.
        :rtype: numbers.Real.
        """

        return self.s()[-1]


class PrjAttitudes(list):

    def __init__(self, atts: List[ProjctAttitude]):

        check_type(atts, "Attitude projections", List)
        for el in atts:
            check_type(el, "Attitude projection", ProjctAttitude)

        super(PrjAttitudes, self).__init__(atts)

    def plot(
            self,
            fig,
            section_length: numbers.Real,
            #vertical_exaggeration: numbers.Real = 1.0,
            plot_addit_params=None,
            color='red'
    ):
        """
        :param fig: the figure in which to plot.
        :type fig:
        :param section_length: the length of the current section.
        :type section_length: numbers.Real.
        :param plot_addit_params:
        :param axes:
        :param section_length:
        :param color:
        :return: the figure.
        :rtype:
        """

        projected_z = [structural_attitude.z for structural_attitude in self if
                       0.0 <= structural_attitude.s <= section_length]

        projected_s = [structural_attitude.s for structural_attitude in self if
                       0.0 <= structural_attitude.s <= section_length]

        projected_ids = [structural_attitude.id for structural_attitude in self if
                         0.0 <= structural_attitude.s <= section_length]

        axes = fig.gca()
        vertical_exaggeration = axes.get_aspect()

        axes.plot(projected_s, projected_z, 'o', color=color)

        # plot segments representing structural data

        for structural_attitude in self:
            if 0.0 <= structural_attitude.s <= section_length:

                structural_segment_s, structural_segment_z = structural_attitude.plot_segment(
                    section_length,
                    vertical_exaggeration)

                fig.gca().plot(structural_segment_s, structural_segment_z, '-', color=color)

        return fig
