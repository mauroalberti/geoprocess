
from typing import List, Optional, Tuple

from array import array

from ..types.utils import check_type
from .elements import AttitudePrjct


class ScalarProfile():
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


class AttitudesPrjct(list):

    def __init__(self):

        super(AttitudesPrjct, self).__init__()

    def add_attitudes(self, atts: List[AttitudePrjct]):

        check_type(atts, "Attitude projections", List)
        for el in atts:
            check_type(el, "Attitude projection", AttitudePrjct)

        self.append(atts)

    def plot(
            self,
            fig,
            section_length: float,
            #vertical_exaggeration: Union[int, float] = 1.0,
            plot_addit_params=None,
            color='red'
    ):
        """
        :param fig: the figure in which to plot.
        :type fig:
        :param section_length: the length of the current section.
        :type section_length: float.
        :param plot_addit_params:
        :param axes:
        :param section_length:
        :param color:
        :return: the figure.
        :rtype:
        """

        print("Fig: {}".format(type(fig)))
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