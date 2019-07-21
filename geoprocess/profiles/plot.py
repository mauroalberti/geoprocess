
from typing import Union, List, Optional


"""
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
"""

from math import sin, cos, sqrt

import numpy as np

import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt


from pygsf.spatial.rasters.geoarray import GeoArray


from .base import *

from ..widgets.qt_tools import qcolor2rgbmpl
from ..widgets.mpl_widget import MplWidget, plot_line, plot_filled_line


colors_addit = ["darkseagreen", "darkgoldenrod", "darkviolet", "hotpink", "powderblue", "yellowgreen",
                "palevioletred",
                "seagreen", "darkturquoise", "beige", "darkkhaki", "red", "yellow", "magenta", "blue", "cyan",
                "chartreuse"]


def plot_profiles(
        profiles: ScalarProfiles,
        aspect: Union[float, int] = 1,
        width: Union[float, int] = 18.5,
        height: Union[float, int] = 10.5):
    """
    Optionally plot a set of profiles with Matplotlib.

    :param profiles: the profiles to plot.
    :type profiles: ScalarProfiles
    :param aspect: the plot aspect.
    :type aspect: Union[float, int].
    :param width: the plot width, in inches. # TOCHECK IF ALWAYS INCHES
    :type width: Union[float, int].
    :param height: the plot height in inches.  # TOCHECK IF ALWAYS INCHES
    :type height: Union[float, int].
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

"""
def plot_profile(profile: LinearProfiler, grid: GeoArray, color: str = "blue", aspect: Union[float, int] = 1, width: float = 18.5, height: float = 10.5):
    '''
    Deprecated. Use 'plot_profiles'.


    :param profile: the linear profile.
    :type: LinearProfile.
    :param grid: the grid storing the elevations.
    :type grid: GeoArray.
    :param color: color.
    :type color: str.
    :param aspect: the profile aspect.
    :type aspect: float, int.
    :param width: the width of the produced figure, in inches.
    :type width: float.
    :param height: the height of the produced figure, in inches.
    :type height: float.
    :return: None.
    '''

    if not isinstance(profile, LinearProfiler):
        return None

    if not isinstance(grid, GeoArray):
        return None

    x = profile.densified_steps()
    y = profile.get_z_values(grid)

    if not y:
        return None

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    ax.set_aspect(aspect)

    ax.plot(x, y)
"""

"""
def plot_topoprofile(topo_profile: TopoProfile, color: str = "blue", aspect: Union[float, int] = 1, width: float = 18.5, height: float = 10.5):
    '''
    Deprecated. Use 'plot_profiles'
    
    Plot a vertical profile given a TopoProfile instance.

    :param topo_profile: TopoProfile instance.
    :type topo_profile: TopoProfile.
    :param color: color.
    :type color: str.
    :param aspect: the profile aspect.
    :type aspect: float, int.
    :param width: the width of the produced figure, in inches.
    :type width: float.
    :param height: the height of the produced figure, in inches.
    :type height: float.
    :return: None.
    '''

    x = topo_profile.profile_s()
    y = topo_profile.elevations()

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)

    ax.set_aspect(aspect)

    ax.plot(x, y)
"""


def define_plot_structural_segment(
        profile_attitude: ProfileAttitude,
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

    h_dist = profile_attitude.sign_hor_dist
    slope_rad = radians(profile_attitude.slope_degr)
    intersection_downward_sense = profile_attitude.dwnwrd_sense
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


def plot_structural_attitudes(
    profile_attitudes: List[Optional[ProfileAttitude]],
    section_length: float,
    fig,
    vertical_exaggeration: Union[int, float] = 1.0,
    plot_addit_params=None,
    color='red'
):
    """

    :param plot_addit_params:
    :param axes:
    :param section_length:
    :param vertical_exaggeration:
    :param profile_attitudes:
    :type profile_attitudes: List[ProfileAttitude].
    :param color:
    :return: the figure.
    :rtype:
    """

    projected_z = [structural_attitude.z for structural_attitude in profile_attitudes if
                   0.0 <= structural_attitude.sign_hor_dist <= section_length]

    projected_s = [structural_attitude.sign_hor_dist for structural_attitude in profile_attitudes if
                   0.0 <= structural_attitude.sign_hor_dist <= section_length]

    """
    projected_ids = [structural_attitude.id for structural_attitude in profile_attitudes if
                     0.0 <= structural_attitude.sign_hor_dist <= section_length]
    """

    fig.gca().plot(projected_s, projected_z, 'o', color=color)

    # plot segments representing structural data

    for structural_attitude in profile_attitudes:
        if 0.0 <= structural_attitude.sign_hor_dist <= section_length:
            structural_segment_s, structural_segment_z = define_plot_structural_segment(structural_attitude,
                                                                                        section_length,
                                                                                        vertical_exaggeration)

            fig.gca().plot(structural_segment_s, structural_segment_z, '-', color=color)


    """
    if plot_addit_params["add_trendplunge_label"] or plot_addit_params["add_ptid_label"]:

        src_dip_dirs = [structural_attitude.src_geol_plane.dd for structural_attitude in
                        profile_attitudes if 0.0 <= structural_attitude.sign_hor_dist <= section_length]
        src_dip_angs = [structural_attitude.src_geol_plane.da for structural_attitude in
                        profile_attitudes if 0.0 <= structural_attitude.sign_hor_dist <= section_length]

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

    #matplotlib.pyplot.show()

    return fig


def plot_projected_line_set(axes, curve_set, labels):

    colors = colors_addit * (int(len(curve_set) / len(colors_addit)) + 1)
    for multiline_2d, label, color in zip(curve_set, labels, colors):
        for line_2d in multiline_2d.lines:
            plot_line(axes, line_2d.x_list, line_2d.y_list, color, name=label)


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

        # plot geological attitudes intersections

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







