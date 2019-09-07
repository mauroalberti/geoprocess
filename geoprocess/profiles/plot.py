
from warnings import warn

from functools import singledispatch, partial


from matplotlib.figure import Figure


from geoprocess.profiles.geoprofiles import *


z_padding = 0.2


def parse_point_on_profile(
        topo_profile: TopographicProfile,
        point: Point
) -> PlotPoint:

    pass


def parse_segment_on_profile(
        topo_profile: TopographicProfile,
        segment: Segment
) -> PlotSegment:

    pass


def parse_intersections(
    topo_profile: TopographicProfile,
    lines_intersections: LinesIntersections
) -> Tuple[List, List]:

    point_intersections = filter(lambda intersection: isinstance(intersection.geom, Point), lines_intersections)
    segment_intersections = filter_segments(lines_intersections)

    plot_points = list(map(partial(parse_point_on_profile, topo_profile=topo_profile), point_intersections))
    plot_segments = list(map(partial(parse_segment_on_profile, topo_profile=topo_profile), segment_intersections))

    return plot_points, plot_segments


@singledispatch
def plot(
    object,
    **kargs
) -> Figure:
    """

    :param object:
    :param kargs:
    :return:
    """

    fig = kargs.get("fig", None)
    aspect = kargs.get("aspect", 1)
    width = kargs.get("width", 18.5)
    height = kargs.get("height", 10.5)

    if fig is None:

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

    else:

        ax = plt.gca()

    return fig


@plot.register(GeoProfile)
def _(
    geoprofile: GeoProfile,
    **kargs
) -> Figure:
    """
    Plot a single geological profile.

    :param geoprofile: the geoprofile to plot
    :type geoprofile: GeoProfile
    :return: the figure.
    :rtype: Figure
    """

    fig = kargs.get("fig", None)
    plot_z_min = kargs.get("plot_z_min", None)
    plot_z_max = kargs.get("plot_z_max", None)

    if plot_z_min is None or plot_z_max is not None:
        z_range = geoprofile.z_max() - geoprofile.z_min()
        plot_z_min = geoprofile.z_min() - z_padding * z_range
        plot_z_max = geoprofile.z_max() + z_padding * z_range

    if fig is None:

        aspect = kargs.get("aspect", 1)
        width = kargs.get("width", 18.5)
        height = kargs.get("height", 10.5)

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)
        ax.set_ylim([plot_z_min, plot_z_max])

    else:

        ax = plt.gca()

    if geoprofile.topo_profile:

        topo_color = kargs.get("topo_color", "blue")

        ax.plot(
            geoprofile.topo_profile.s_arr(),
            geoprofile.topo_profile.z_arr(),
            color=topo_color
        )

    if geoprofile.attitudes:

        attits = geoprofile.attitudes

        attitude_color = kargs.get("attitude_color", "red")
        section_length = geoprofile.length_2d()

        projected_z = [structural_attitude.z for structural_attitude in attits if
                       0.0 <= structural_attitude.s <= section_length]

        projected_s = [structural_attitude.s for structural_attitude in attits if
                       0.0 <= structural_attitude.s <= section_length]

        projected_ids = [structural_attitude.id for structural_attitude in attits if
                         0.0 <= structural_attitude.s <= section_length]

        axes = fig.gca()
        vertical_exaggeration = axes.get_aspect()

        axes.plot(projected_s, projected_z, 'o', color=attitude_color)

        # plot segments representing structural data

        for structural_attitude in attits:
            if 0.0 <= structural_attitude.s <= section_length:

                structural_segment_s, structural_segment_z = structural_attitude.create_segment_for_plot(
                    section_length,
                    vertical_exaggeration)

                fig.gca().plot(structural_segment_s, structural_segment_z, '-', color=attitude_color)

    if geoprofile.lines_intersections:

        if not geoprofile.topo_profile:

            warn('Topographic profile is not defined, so intersections cannot be plotted')

        else:

            intersection_points, intersection_segments = parse_intersections(
                geoprofile.topo_profile,
                geoprofile.lines_intersections
            )

    if geoprofile.polygons_intersections:

        pass


    return fig


@plot.register(GeoProfileSet)
def _(
    geoprofiles: GeoProfileSet,
    **kargs
) -> Figure:
    """
    Plot a set of geological profiles.

    :param geoprofiles: the geoprofiles to plot
    :type geoprofiles: GeoProfiles
    :return: the figure.
    :rtype: Figure
    """

    num_subplots = geoprofiles.num_profiles()

    z_range = geoprofiles.z_max() - geoprofiles.z_min()
    plot_z_min = geoprofiles.z_min() - z_range * z_padding
    plot_z_max = geoprofiles.z_max() + z_range * z_padding

    for ndx in range(num_subplots):
        geoprofile = geoprofiles.extract_geoprofile(ndx)
        plot(
            geoprofile,
            plot_z_min=plot_z_min,
            plot_z_max=plot_z_max
        )