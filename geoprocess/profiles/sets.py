
from pygsf.spatial.vectorial.geometries import *

from ..widgets.mpl_widget import plot_line, plot_filled_line
from ..widgets.qt_tools import qcolor2rgbmpl

from .elements import *
from .chains import *


class DrillsProjcts:

    pass

'''
class TopographicProfile(list):
    """
    Class storing a set (one or more) of scalar profiles.

    They share a set of location distances from the profile start.
    Per se they are not required to express a linear profile.

    All the profiles store double values (scalar).


    """

    def __init__(self):

        super(TopographicProfile, self).__init__()

    def num_profiles(self) -> int:
        """
        Return the number of available profiles.

        :return: number of available profiles.
        :rtype: int.
        """

        return len(self)
'''

class ScalarsPrevious:
    """

    Deprecated. Use class TopographicProfile.

    Class storing a vertical topographic profile element.
    """

    def __init__(self, line: Line):
        """
        Instantiates a topographic profile object.

        :param line: the topographic profile line instance.
        :type line: Line.
        """

        if not isinstance(line, Line):
            raise Exception("Input must be a Line instance")

        if line.length_2d() == 0.0:
            raise Exception("Input line length is zero")

        self._line = line

        self.horiz_dist_values, self.dist_3d_values, self.dir_slopes_rads = profile_parameters(self._line)

    def line(self) -> Line:
        """
        Returns the topographic profile line.

        :return: the line of the profile.
        :rtype: Line
        """

        return self._line

    def start_pt(self) -> Optional[Point]:
        """
        Returns the first point of a profile.

        :return: the profile first point.
        :rtype: Optional[Point].
        """

        return self._line.start_pt()

    def end_pt(self) -> Optional[Point]:
        """
        Returns the last point of a profile.

        :return: the profile last point.
        :rtype: Optional[Point].
        """

        return self._line.end_pt()

    def profile_s(self) -> List[numbers.Real]:
        """
        Returns the incremental 2D lengths of a profile.

        :return: the incremental 2D lengths.
        :rtype: list of numbers.Real values.
        """

        return list(itertools.accumulate(self.horiz_dist_values))

    def profile_length(self) -> numbers.Real:
        """
        Returns the length of the profile.

        :return: length of profile.
        :rtype: numbers.Real.
        """

        return self._line.length_2d()

    def profile_length_3d(self) -> numbers.Real:
        """
        Returns the 3D length of the profile.

        :return: 3D length of profile.
        :rtype: numbers.Real.
        """

        return self._line.length_3d()

    def elevations(self) -> List[numbers.Real]:
        """
        Returns the elevations of the profile.

        :return: the elevations.
        :rtype: list of floats.
        """

        return self._line.z_list()

    def elev_stats(self) -> Dict:
        """
        Calculates profile elevation statistics.

        :return: the elevation statistic values.
        :rtype: Dict.
        """

        return self._line.z_stats()

    def slopes(self) -> List[Optional[numbers.Real]]:
        """
        Returns the slopes of a topographic profile.

        :return: slopes.
        :rtype: list of slope values.
        """

        return self._line.slopes()

    def abs_slopes(self) -> List[Optional[numbers.Real]]:
        """
        Returns the absolute slopes of a topographic profile.

        :return: absolute slopes.
        :rtype: list of slope values.
        """

        return self._line.abs_slopes()

    def slopes_stats(self) -> Dict:
        """
        Calculates profile directional slopes statistics.

        :return: the slopes statistic values.
        :rtype: Dict.
        """

        return self._line.slopes_stats()

    def absslopes_stats(self) -> Dict:
        """
        Calculates profile absolute slopes statistics.

        :return: the absolute slopes statistic values.
        :rtype: Dict.
        """

        return self._line.abs_slopes_stats()


class LinesIntersections:

    pass


class TracesProjections:

    pass


class PolygonsIntersections:

    pass


def profile_parameters(profile: Line) -> Tuple[List[numbers.Real], List[numbers.Real], List[numbers.Real]]:
    """
    Calculates profile parameters for polar projections source datasets.

    :param profile: the profile line.
    :type profile: Line.
    :return: three profile parameters: horizontal distances, 3D distances, directional slopes
    :rtype: Tuple of three floats lists.
    """

    # calculate 3D distances between consecutive points

    if profile.epsg() == 4326:

        # convert original values into ECEF values (x, y, z, time in ECEF global coordinate system)
        ecef_ln = profile.wgs842ecef()

        dist_3d_values = ecef_ln.step_lengths_3d()

    else:

        dist_3d_values = profile.step_lengths_3d()

    # calculate delta elevations between consecutive points

    delta_elev_values = profile.step_delta_z()

    # calculate slope along section

    dir_slopes_rads = []
    for delta_elev, dist_3D in zip(delta_elev_values, dist_3d_values):
        if dist_3D == 0.0:
            if delta_elev == 0.0:
                slope_rads = 0.0
            elif delta_elev < 0.0:
                slope_rads = - 0.5 * pi
            else:
                slope_rads = 0.5 * pi
        else:
            slope_rads = asin(delta_elev / dist_3D)

        dir_slopes_rads.append(slope_rads)

    # calculate horizontal distance along section

    horiz_dist_values = []
    for slope_rads, dist_3D in zip(dir_slopes_rads, dist_3d_values):

        horiz_dist_values.append(dist_3D * cos(slope_rads))

    return horiz_dist_values, dist_3d_values, dir_slopes_rads



    """
    if plot_addit_params["add_trendplunge_label"] or plot_addit_params["add_ptid_label"]:

        src_dip_dirs = [structural_attitude.src_geol_plane.dd for structural_attitude in
                        profile_attitudes if 0.0 <= structural_attitude.s <= section_length]
        src_dip_angs = [structural_attitude.src_geol_plane.da for structural_attitude in
                        profile_attitudes if 0.0 <= structural_attitude.s <= section_length]

        for rec_id, src_dip_dir, src_dip_ang, s, z in zip(projected_ids, src_dip_dirs, src_dip_angs, projected_s,
                                                          projected_z):

            if plot_addit_params["add_trendplunge_label"] and plot_addit_params["add_ptid_label"]:
                label = "%s-%03d/%02d" % (rec_id, src_dip_dir, src_dip_ang)
            elif plot_addit_params["add_ptid_label"]:
                label = "%s" % rec_id
            elif plot_addit_params["add_trendplunge_label"]:
                label = "%03d/%02d" % (src_dip_dir, src_dip_ang)

            fig.gca().annotate(label, (s + 15, z + 15))
    """

    return fig


def plot_profile_lines_intersection_points(axes, profile_lines_intersection_points):

    for s, pt3d, intersection_id, color in profile_lines_intersection_points:
        axes.plot(s, pt3d.z, 'o', color=color)
        if str(intersection_id).upper() != "NULL" or str(intersection_id) != '':
            axes.annotate(str(intersection_id), (s + 25, pt3d.z + 25))


def plot_profile_polygon_intersection_line(plot_addit_params, axes, intersection_line_value):

    classification, line3d, s_list = intersection_line_value
    z_list = [pt3d.z for pt3d in line3d.pts]

    if plot_addit_params["polygon_class_colors"] is None:
        color = "red"
    else:
        color = plot_addit_params["polygon_class_colors"][str(classification)]

    plot_line(axes, s_list, z_list, color, linewidth=3.0, name=classification)


def plot_profiles(
        profiles: TopographicProfile,
        aspect: numbers.Real = 1,
        width: numbers.Real = 18.5,
        height: numbers.Real = 10.5):
    """
    Deprecated. Use inner method of Geoprofile class.

    Optionally plot a set of profiles with Matplotlib.

    :param profiles: the profiles to plot.
    :type profiles: TopographicProfile
    :param aspect: the plot aspect.
    :type aspect: numbers.Real.
    :param width: the plot width, in inches. # TOCHECK IF ALWAYS INCHES
    :type width: numbers.Real.
    :param height: the plot height in inches.  # TOCHECK IF ALWAYS INCHES
    :type height: numbers.Real.
    :return: the figure.
    :rtype:
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    ax.set_aspect(aspect)

    s = profiles.s()

    for z in profiles.zs():
        if z:
            ax.plot(s, z)

    return fig


def plot_projected_line_set(axes, curve_set, labels):

    colors = colors_addit * (int(len(curve_set) / len(colors_addit)) + 1)
    for multiline_2d, label, color in zip(curve_set, labels, colors):
        for line_2d in multiline_2d.lines:
            plot_line(axes, line_2d.x_list, line_2d.y_list, color, name=label)


def plot_topo_profile_lines(grid_spec, ndx_subplot, topo_type, plot_x_range, plot_y_range, filled_choice):


    topoline_colors = plot_params['elev_lyr_colors']
    topoline_visibilities = plot_params['visible_elev_lyrs']

    axes = create_axes(
        profile_window,
        plot_x_range,
        plot_y_range)

    if plot_params['invert_xaxis']:
        axes.invert_xaxis()

    if topo_type == 'elevation':
        ys = geoprofile.profiles_zs()
        plot_y_min = plot_y_range[0]
    else:
        if plot_params['plot_slope_absolute']:
            ys = geoprofile.abs_slopes()
        else:
            ys = geoprofile.slopes()
        plot_y_min = 0.0

    s = geoprofile.profiles_svals()[0]

    for y, topoline_color, topoline_visibility in zip(ys, topoline_colors, topoline_visibilities):

        if topoline_visibility:

            if filled_choice:
                plot_filled_line(
                    axes,
                    s,
                    y,
                    plot_y_min,
                    qcolor2rgbmpl(topoline_color))

            plot_line(
                axes,
                s,
                y,
                qcolor2rgbmpl(topoline_color))

    return axes


def create_axes(profile_window, plot_x_range, plot_y_range):

    print("Creating axes")

    x_min, x_max = plot_x_range
    y_min, y_max = plot_y_range
    axes = profile_window.canvas.fig.add_subplot(grid_spec[ndx_subplot])
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)

    axes.grid(True)

    print(" - created")

    return axes


colors_addit = ["darkseagreen", "darkgoldenrod", "darkviolet", "hotpink", "powderblue", "yellowgreen",
                "palevioletred",
                "seagreen", "darkturquoise", "beige", "darkkhaki", "red", "yellow", "magenta", "blue", "cyan",
                "chartreuse"]