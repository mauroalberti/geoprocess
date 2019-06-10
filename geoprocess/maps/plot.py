
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pygsf.spatial.vectorial.geometries import Line
from pygsf.spatial.rasters.geoarray import GeoArray


def plot_grid(geo_array: GeoArray, level_ndx: int = 0, cmap="jet", width: float = 18.5, height: float = 10.5) -> Figure:
    """
    Plot a grid with geographic coordinates.

    :param geo_array: the input GeoArray instance.
    :type geo_array: GeoArray.
    :param level_ndx: the band index. Starts at 0.
    :type level_ndx: int.
    :param cmap: the colormap to apply.
    :type cmap: basestring.
    :param width: figure width in inches.
    :type width: float.
    :param height: figure height in inches.
    :type height: float.
    :return: Figure.
    :raise: Exception when grid has rotation.

    """

    if geo_array.has_rotation:
        raise Exception("Source grid has rotation.")

    top_left_geogcoord, top_right_geogcoord, btm_right_geogcoord, btm_left_geogcoord = geo_array.band_corners_geogcoords(level_ndx=level_ndx)
    geo_extent = [
        btm_left_geogcoord[0], top_right_geogcoord[0],
        btm_left_geogcoord[1], top_right_geogcoord[1]]

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    ax.imshow(geo_array.level(level_ndx=level_ndx), extent=geo_extent, cmap=cmap)

    return fig


def plot_line(fig: Figure, line: Line) -> None:
    """
    Plot a line.

    :param ax: the figure in which to plot the line.
    :type ax: Figure.
    :param line: the line to plot.
    :type line: Line.
    :return: None.
    """

    fig.gca().plot(line.x_list, line.y_list, '-')
    return fig

