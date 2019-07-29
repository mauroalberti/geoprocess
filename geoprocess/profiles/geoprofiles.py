
from typing import Optional

from matplotlib import gridspec
import matplotlib.pyplot as plt


from pygsf.geology.orientations import *

from .chains import *
from ..types.utils import check_type
from ..widgets.mpl_widget import MplWidget

from .sets import *


class GeoProfile:
    """
    Class representing the topographic and geological elements
    embodying a single geological profile.
    """

    def __init__(self):
        """

        """

        self._topo_profile = None
        self._attitudes = None
        self.lines_intersections = None
        self.polygons_intersections = None
        self.traces_projections = None
        self.geosurfaces_ids = None

    @property
    def topo_profile(self):
        """

        :return:
        """

        return self._topo_profile

    @topo_profile.setter
    def topo_profile(self,
        scalars_inters: TopographicProfile):
        """

        :param scalars_inters: the scalar values profiles.
        :type scalars_inters: TopographicProfile.
        :return:

        """

        check_type(scalars_inters, "Scalars intersections", TopographicProfile)
        self._topo_profile = scalars_inters

    def clear_topo_profile(self):
        """

        :return:
        """

        self._topo_profile = None

    def plot_topo_profile(self,
          aspect: Union[float, int] = 1,
          width: Union[float, int] = 18.5,
          height: Union[float, int] = 10.5,
          color="blue"):
        """
        Plot a set of profiles with Matplotlib.

        :param aspect: the plot aspect.
        :type aspect: Union[float, int].
        :param width: the plot width, in inches. # TOCHECK IF ALWAYS INCHES
        :type width: Union[float, int].
        :param height: the plot height in inches.  # TOCHECK IF ALWAYS INCHES
        :type color: the color.
        :param color: str.
        :type height: Union[float, int].
        :return: the figure.
        :rtype:
        """

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

        if self._topo_profile:
            ax.plot(
                self._topo_profile.s(),
                self._topo_profile.z(),
                color=color
            )

        self.fig = fig

    @property
    def attitudes(self):
        """

        :return:
        """

        return self._attitudes

    @attitudes.setter
    def attitudes(self,
        prj_attitudes: PrjAttitudes):
        """
        Set the projected _attitudes content.

        :param prj_attitudes: projected _attitudes.
        :type prj_attitudes: PrjAttitudes.
        :return:
        """

        check_type(prj_attitudes, "Projected _attitudes", List)
        for el in prj_attitudes:
            check_type(el, "Projected attitude", ProjctAttitude)

        self._attitudes = prj_attitudes

    def clear_attitudes(self):
        """
        Clear projected _attitudes content.

        :return:
        """

        self._attitudes = None

    def plot_attitudes(self, color="red"):
        """

        :return:
        """

        self.fig = self._attitudes.plot(
            self.fig,
            self.length_2d(),
            color=color
        )

    def plot(self,
             topo_profile=True,
             attitudes=True,
             line_intersections=True,
             polygon_intersections=True,
             line_projections=True,
             topo_profile_color="blue",
             attitudes_color="red",
             line_intersections_color="orange",
             aspect: Union[float, int] = 1,
             width: Union[float, int] = 18.5,
             height: Union[float, int] = 10.5,
             ):
        """

        :param topo_profile:
        :param attitudes:
        :param line_intersections:
        :param polygon_intersections:
        :param line_projections:
        :return:
        """

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

        if topo_profile and self._topo_profile:
            ax.plot(
                self._topo_profile.s(),
                self._topo_profile.z(),
                color=topo_profile_color
            )

        if attitudes and self._attitudes:
            self._attitudes.plot(
                fig,
                self.length_2d(),
                color=attitudes_color
            )

    def add_intersections_pts(self, intersection_list):
        """

        :param intersection_list:
        :return:
        """

        self.lines_intersections += intersection_list

    def add_intersections_lines(self, formation_list, intersection_line3d_list, intersection_polygon_s_list2):
        """

        :param formation_list:
        :param intersection_line3d_list:
        :param intersection_polygon_s_list2:
        :return:
        """

        self.polygons_intersections = zip(formation_list, intersection_line3d_list, intersection_polygon_s_list2)

    def profiles_svals(self) -> List[List[float]]:
        """
        Returns the list of the s values for the profiles.

        :return: list of the s values.
        :rtype
        """

        return [topoprofile.profile_s() for topoprofile in self._topo_profile]

    def profiles_zs(self) -> List[float]:
        """
        Returns the elevations of the profiles.

        :return: the elevations.
        :rtype: list of float values.
        """

        return [topoprofile.elevations() for topoprofile in self._topo_profile]

    def length_2d(self) -> float:
        """
        Returns the 2D length of the profile.

        :return: the 2D profile length.
        :rtype: float.
        """

        return self._topo_profile.profile_length()

    def profiles_lengths_3d(self) -> List[float]:
        """
        Returns the 3D lengths of the profiles.

        :return: the 3D profiles lengths.
        :rtype: list of float values.
        """

        return [topoprofile.profile_length_3d() for topoprofile in self._topo_profile]

    def minmax_length_2d(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns the maximum 2D length of profiles.

        :return: the minimum and maximum profiles lengths.
        :rtype: a pair of optional float values.
        """

        lengths = self.length_2d()

        if lengths:
            return min(lengths), max(lengths)
        else:
            return None, None

    def max_length_2d(self) -> Optional[float]:
        """
        Returns the maximum 2D length of profiles.

        :return: the maximum profiles lengths.
        :rtype: an optional float value.
        """

        lengths = self.length_2d()

        if lengths:
            return max(lengths)
        else:
            return None

    def topoprofiles_params(self):
        """

        :return:
        """

        topo_lines = [topo_profile.line for topo_profile in self._topo_profile]
        max_s = max(map(lambda line: line.length_2d(), topo_lines))
        min_z = min(map(lambda line: line.z_min(), topo_lines))
        max_z = max(map(lambda line: line.z_max(), topo_lines))

        return max_s, min_z, max_z

    def elev_stats(self) -> List[Dict]:
        """
        Returns the elevation statistics for the topographic profiles.

        :return: elevation statistics for the topographic profiles.
        :rtype: list of dictionaries.
        """

        return [topoprofile.elev_stats() for topoprofile in self._topo_profile]

    def slopes(self) -> List[List[Optional[float]]]:
        """
        Returns a list of the slopes of the topographic profiles in the geoprofile.

        :return: list of the slopes of the topographic profiles.
        :rtype: list of list of topographic slopes.
        """

        return [topoprofile.slopes() for topoprofile in self._topo_profile]

    def abs_slopes(self) -> List[List[Optional[float]]]:
        """
        Returns a list of the absolute slopes of the topographic profiles in the geoprofile.

        :return: list of the absolute slopes of the topographic profiles.
        :rtype: list of list of topographic absolute slopes.
        """

        return [topoprofile.abs_slopes() for topoprofile in self._topo_profile]

    def slopes_stats(self) -> List[Dict]:
        """
        Returns the directional slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.slopes_stats() for topoprofile in self._topo_profile]

    def absslopes_stats(self) -> List[Dict]:
        """
        Returns the absolute slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.absslopes_stats() for topoprofile in self._topo_profile]

    def min_z_plane_attitudes(self):
        """

        :return:
        """

        # TODO:  manage case for possible nan p_z values
        return min([plane_attitude.pt_3d.p_z for plane_attitude_set in self._attitudes for plane_attitude in
                    plane_attitude_set if 0.0 <= plane_attitude.sign_hor_dist <= self.max_s()])

    def max_z_plane_attitudes(self):
        """

        :return:
        """

        # TODO:  manage case for possible nan p_z values
        return max([plane_attitude.pt_3d.p_z for plane_attitude_set in self._attitudes for plane_attitude in
                    plane_attitude_set if 0.0 <= plane_attitude.sign_hor_dist <= self.max_s()])

    def min_z_curves(self):
        """

        :return:
        """

        return min([pt_2d.p_y for multiline_2d_list in self.traces_projections for multiline_2d in multiline_2d_list for line_2d in
                    multiline_2d.lines for pt_2d in line_2d.pts if 0.0 <= pt_2d.p_x <= self.max_s()])

    def max_z_curves(self):
        """

        :return:
        """

        return max([pt_2d.p_y for multiline_2d_list in self.traces_projections for multiline_2d in multiline_2d_list for line_2d in
                    multiline_2d.lines for pt_2d in line_2d.pts if 0.0 <= pt_2d.p_x <= self.max_s()])

    def mins_z_topo(self) -> List[float]:
        """
        Returns a list of elevation minimums in the topographic profiles.

        :return: the elevation minimums.

        """

        return [topo_profile.elev_stats["min"] for topo_profile in self._topo_profile]

    def maxs_z_topo(self) -> List[float]:
        """
        Returns a list of elevation maximums in the topographic profiles.

        :return:
        """

        return [topo_profile.elev_stats["max"] for topo_profile in self._topo_profile]

    def min_z_topo(self) -> Optional[float]:
        """
        Returns the minimum elevation value in the topographic profiles.

        :return: the minimum elevation value in the profiles.
        :rtype: optional float.
        """

        mins_z = self.mins_z_topo()
        if mins_z:
            return min(mins_z)
        else:
            return None

    def max_z_topo(self) -> Optional[float]:
        """
        Returns the maximum elevation value in the topographic profiles.

        :return: the maximum elevation value in the profiles.
        :rtype: optional float.
        """

        maxs_z = self.maxs_z_topo()
        if maxs_z:
            return max(maxs_z)
        else:
            return None

    def natural_elev_range(self) -> Tuple[float, float]:
        """
        Returns the elevation range of the profiles.

        :return: minimum and maximum values of the considered topographic profiles.
        :rtype: tuple of two floats.
        """

        return self.min_z_topo(), self.max_z_topo()

    def min_z(self):
        """

        :return:
        """

        min_z = self.min_z_topo()

        if len(self._attitudes) > 0:
            min_z = min([min_z, self.min_z_plane_attitudes()])

        if len(self.traces_projections) > 0:
            min_z = min([min_z, self.min_z_curves()])

        return min_z

    def max_z(self):
        """

        :return:
        """

        max_z = self.max_z_topo()

        if len(self._attitudes) > 0:
            max_z = max([max_z, self.max_z_plane_attitudes()])

        if len(self.traces_projections) > 0:
            max_z = max([max_z, self.max_z_curves()])

        return max_z

    def add_plane_attitudes(self, plane_attitudes):
        """

        :param plane_attitudes:
        :return:
        """

        self._attitudes.append(plane_attitudes)

    def add_curves(self, lMultilines, lIds):
        """

        :param lMultilines:
        :param lIds:
        :return:
        """

        self.traces_projections.append(lMultilines)
        self.geosurfaces_ids.append(lIds)


class GeoProfilesSet:
    """
    Represents a set of Geoprofile elements,
    stored as a list
    """

    def __init__(self):

        self._geoprofiles = []  # a list of GeoProfile instances

    @property
    def geoprofiles(self):
        """

        :return:
        """

        return self._geoprofiles

    @property
    def geoprofiles_num(self) -> int:
        """
        Returns the number of geoprofiles.

        :return: number of geoprofiles.
        :rtype: int.
        """

        return len(self.geoprofiles)

    def geoprofile(self, ndx):
        """

        :param ndx:
        :return:
        """

        return self._geoprofiles[ndx]

    def append(self, geoprofile):
        """

        :param geoprofile:
        :return:
        """

        self._geoprofiles.append(geoprofile)

    def insert(self, ndx, geoprofile):
        """

        :param ndx:
        :param geoprofile:
        :return:
        """

        self._geoprofiles.insert(ndx, geoprofile)

    def move(self, ndx_init, ndx_final):
        """

        :param ndx_init:
        :param ndx_final:
        :return:
        """

        geoprofile = self._geoprofiles.pop(ndx_init)
        self.insert(ndx_final, geoprofile)

    def move_up(self, ndx):
        """

        :param ndx:
        :return:
        """

        self.move(ndx, ndx -1)

    def move_down(self, ndx):
        """

        :param ndx:
        :return:
        """

        self.move(ndx, ndx + 1)

    def remove(self, ndx):
        """

        :param ndx:
        :return:
        """

        _ = self._geoprofiles.pop(ndx)


"""
class TopoProfiles(object):

    A set of topographic profiles created from a
    single planar profile trace and a set of source elevations
    data (for instance, a pair of dems).

    def __init__(self,
                 crs_authid: str,
                 profile_source: str,
                 source_names: List[str],
                 xs: List[float],
                 ys: List[float],
                 zs: List[List[float]],
                 times: List[float],
                 inverted: bool):

        self.crs_authid = crs_authid
        self.profile_source = profile_source
        self.source_names = source_names
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.times = times
        self.inverted = inverted

        ###

        if self.crs_authid == "EPSG:4326":

            self.profile_s, self.profiles_s3ds, self.profiles_dirslopes = self.params_polar()

        else:
            self.profile_s = np.asarray(source_trace.incremental_length_2d())

            self.profiles_dirslopes = [np.asarray(line.slopes()) for line in topo_lines]

            self.profiles_s3ds = [np.asarray(line.incremental_length_3d()) for line in topo_lines]

    @property
    def num_pnts(self) -> int:

        return len(self.xs)

    def max_s(self) -> float:

        return self.profile_s[-1]

    def min_z(self) -> float:

        return float(min([np.nanmin(prof_elev) for prof_elev in self.profiles_elevs]))

    def max_z(self) -> float:

        return float(max([np.nanmax(prof_elev) for prof_elev in self.profiles_elevs]))

    @property
    def absolute_slopes(self) -> List[np.ndarray]:

        return [np.fabs(prof_dirslopes) for prof_dirslopes in self.profiles_dirslopes]

    @property
    def statistics_elev(self) -> List(Dict):

        return [get_statistics(prof_elev) for prof_elev in self.profiles_elevs]

    @property
    def statistics_dirslopes(self) -> List(Dict):

        return [get_statistics(prof_dirslopes) for prof_dirslopes in self.profiles_dirslopes]

    @property
    def statistics_slopes(self) -> List(Dict):

        return [get_statistics(prof_abs_slopes) for prof_abs_slopes in self.absolute_slopes]
"""


def plot_geoprofiles(geoprofiles, plot_addit_params, slope_padding=0.2):

    # extract/define plot parameters

    plot_params = geoprofiles.plot_params

    set_vertical_exaggeration = plot_params["set_vertical_exaggeration"]
    vertical_exaggeration = plot_params['vertical_exaggeration']

    plot_height_choice = plot_params['plot_height_choice']
    plot_slope_choice = plot_params['plot_slope_choice']

    if plot_height_choice:
        # defines plot min and max values
        plot_z_min = plot_params['plot_min_elevation_user']
        plot_z_max = plot_params['plot_max_elevation_user']

    # populate the plot

    profile_window = MplWidget('Profile')

    num_subplots = (plot_height_choice + plot_slope_choice)*geoprofiles.geoprofiles_num
    grid_spec = gridspec.GridSpec(num_subplots, 1)

    ndx_subplot = -1
    for ndx in range(geoprofiles.geoprofiles_num):

        geoprofile = geoprofiles.geoprofile(ndx)
        plot_s_min, plot_s_max = 0, geoprofile.max_length_2d()

        # if slopes to be calculated and plotted
        if plot_slope_choice:

            # defines slope value lists and the min and max values
            if plot_params['plot_slope_absolute']:
                slopes = geoprofile.abs_slopes()
            else:
                slopes = geoprofile.slopes()

            profiles_slope_min = np.nanmin(np.array(list(map(np.nanmin, slopes))))
            profiles_slope_max = np.nanmax(np.array(list(map(np.nanmax, slopes))))

            delta_slope = profiles_slope_max - profiles_slope_min
            plot_slope_min = profiles_slope_min - delta_slope * slope_padding
            plot_slope_max = profiles_slope_max + delta_slope * slope_padding

        # plot topographic profile elevations

        if plot_height_choice:
            ndx_subplot += 1
            axes_elevation = plot_topo_profile_lines(
                grid_spec,
                ndx_subplot,
                'elevation',
                (plot_s_min, plot_s_max),
                (plot_z_min, plot_z_max),
                plot_params['filled_height'])
            if set_vertical_exaggeration:
                axes_elevation.set_aspect(vertical_exaggeration)
            axes_elevation.set_anchor('W')  # align left

        # plot topographic profile slopes

        if plot_slope_choice:
            ndx_subplot += 1
            axes_slopes = plot_topo_profile_lines(
                grid_spec,
                ndx_subplot,
                'slope',
                (plot_s_min, plot_s_max),
                (plot_slope_min, plot_slope_max),
                plot_params['filled_slope'])
            axes_slopes.set_anchor('W')  # align left

        # plot geological outcrop intersections

        if len(geoprofile.outcrops) > 0:
            for line_intersection_value in geoprofile.outcrops:
                plot_profile_polygon_intersection_line(plot_addit_params,
                                                       axes_elevation,
                                                       line_intersection_value)

        # plot geological _attitudes intersections

        if len(geoprofile.geoplane_attitudes) > 0:
            for plane_attitude_set, color in zip(geoprofile.geoplane_attitudes, plot_addit_params["plane_attitudes_colors"]):
                plot_structural_attitudes(plot_addit_params,
                                          axes_elevation,
                                          plot_s_max,
                                          vertical_exaggeration,
                                          plane_attitude_set,
                                          color)

        # plot geological traces projections

        if len(geoprofile.geosurfaces) > 0:
            for curve_set, labels in zip(geoprofile.geosurfaces, geoprofile.geosurfaces_ids):
                plot_projected_line_set(axes_elevation,
                                        curve_set,
                                        labels)

        # plot line-profile intersections

        if len(geoprofile.lineaments) > 0:
            plot_profile_lines_intersection_points(axes_elevation,
                                                   geoprofile.lineaments)

    profile_window.canvas.fig.tight_layout()
    profile_window.canvas.draw()

    return profile_window