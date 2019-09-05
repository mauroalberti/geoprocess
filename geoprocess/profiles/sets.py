
from typing import Optional, Tuple, Dict

from math import pi, asin


from pygsf.spatial.vectorial.geometries import Line

from ..widgets.mpl_widget import plot_line, plot_filled_line
from ..widgets.qt_tools import qcolor2rgbmpl

from .chains import *


class TopographicProfileSet(list):
    """

    Class storing a set of topographic profiles.
    """

    def __init__(self, topo_profiles_set: List[TopographicProfile]):
        """
        Instantiates a topographic profile set.

        :param topo_profiles_set: the topographic profile set.
        :type topo_profiles_set: List[TopographicProfile].
        """

        check_type(topo_profiles_set, "Topographic profiles set", List)
        for el in topo_profiles_set:
            check_type(el, "Topographic profile", TopographicProfile)

        super(TopographicProfileSet, self).__init__(topo_profiles_set)

    def s_min(self) -> Optional[numbers.Real]:
        """
        Returns the minimum s value in the topographic profiles.

        :return: the minimum s value in the profiles.
        :rtype: optional numbers.Real.
        """

        return min([prof.s_min() for prof in self])

    def s_max(self) -> Optional[numbers.Real]:
        """
        Returns the maximum s value in the topographic profiles.

        :return: the maximum s value in the profiles.
        :rtype: optional numbers.Real.
        """

        return max([prof.s_max() for prof in self])

    def z_min(self) -> Optional[numbers.Real]:
        """
        Returns the minimum elevation value in the topographic profiles.

        :return: the minimum elevation value in the profiles.
        :rtype: optional numbers.Real.
        """

        return min([prof.z_min() for prof in self])

    def z_max(self) -> Optional[numbers.Real]:
        """
        Returns the maximum elevation value in the topographic profiles.

        :return: the maximum elevation value in the profiles.
        :rtype: optional numbers.Real.
        """

        return max([prof.z_max() for prof in self])

    def natural_elev_range(self) -> Tuple[numbers.Real, numbers.Real]:
        """
        Returns the elevation range of the profiles.

        :return: minimum and maximum values of the considered topographic profiles.
        :rtype: tuple of two floats.
        """

        return self.z_min(), self.z_max()

    def topoprofiles_params(self):
        """

        :return:
        """

        return self.s_min(), self.s_max(), self.z_min(), self.z_max()

    def elev_stats(self) -> List[Dict]:
        """
        Returns the elevation statistics for the topographic profiles.

        :return: elevation statistics for the topographic profiles.
        :rtype: list of dictionaries.
        """

        return [topoprofile.elev_stats() for topoprofile in self]

    def slopes(self) -> List[List[Optional[numbers.Real]]]:
        """
        Returns a list of the slopes of the topographic profiles in the geoprofile.

        :return: list of the slopes of the topographic profiles.
        :rtype: list of list of topographic slopes.
        """

        return [topoprofile.slopes() for topoprofile in self]

    def abs_slopes(self) -> List[List[Optional[numbers.Real]]]:
        """
        Returns a list of the absolute slopes of the topographic profiles in the geoprofile.

        :return: list of the absolute slopes of the topographic profiles.
        :rtype: list of list of topographic absolute slopes.
        """

        return [topoprofile.abs_slopes() for topoprofile in self]

    def slopes_stats(self) -> List[Dict]:
        """
        Returns the directional slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.slopes_stats() for topoprofile in self]

    def absslopes_stats(self) -> List[Dict]:
        """
        Returns the absolute slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.absslopes_stats() for topoprofile in self]


class AttitudesSet(list):
    """

    Class storing a set of topographic profiles.
    """

    def __init__(self, attitudes_set: List[Attitude]):
        """
        Instantiates an attitudes set.

        :param attitudes_set: the attitudes set.
        :type attitudes_set: List[Attitude].
        """

        check_type(attitudes_set, "Attitude set", List)
        for el in attitudes_set:
            check_type(el, "Attitude", Attitude)

        super(AttitudesSet, self).__init__(attitudes_set)

    # inherited - TO CHECK

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


class LinesIntersectionsSet(list):
    """

    Class storing a set of topographic profiles.
    """

    def __init__(self, line_intersection_set: List[LinesIntersections]):
        """
        Instantiates an lines intersections set.

        :param line_intersection_set: the lines intersections set.
        :type line_intersection_set: List[LinesIntersections].
        """

        check_type(line_intersection_set, "Line intersections set", List)
        for el in line_intersection_set:
            check_type(el, "Line intersections", LinesIntersections)

        super(LinesIntersectionsSet, self).__init__(line_intersection_set)


class PolygonsIntersectionsSet(list):
    """

    Class storing a set of topographic profiles.
    """

    def __init__(self, polygons_intersections_set: List[PolygonsIntersections]):
        """
        Instantiates a polygons intersections set.

        :param polygons_intersections_set: the polygons intersections set.
        :type polygons_intersections_set: List[PolygonsIntersections].
        """

        check_type(polygons_intersections_set, "PolygonsIntersections set", List)
        for el in polygons_intersections_set:
            check_type(el, "Polygons intersections", PolygonsIntersections)

        super(PolygonsIntersectionsSet, self).__init__(polygons_intersections_set)


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

    s = profiles.s_arr()

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