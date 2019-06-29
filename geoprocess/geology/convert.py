
from typing import List, Tuple

from geopandas import GeoDataFrame

from pygsf.spatial.vectorial.geometries import Point
from pygsf.geology.orientations import Plane


def extract_georeferenced_attitudes(geodataframe: GeoDataFrame, dip_dir_fldnm: str, dip_ang_fldnm: str) -> List[Tuple[Point, Plane]]:
    """
    Extracts the georeferenced attitudes from a geopandas GeoDataFrame instance.

    :param geodataframe: the source geodataframe.
    :type geodataframe: GeoDataFrame.
    :param dip_dir_fldnm: the name of the dip direction field in the geodataframe.
    :type dip_dir_fldnm: basestring.
    :param dip_ang_fldnm: the name of the dip angle field in the geodataframe.
    :type dip_ang_fldnm: basestring.
    :return: a collection of Point and Plane values, one for each source record.
    :rtype: List[Tuple[Point, Plane]]
    """

    crs = geodataframe.crs['init']
    if crs.startswith("epsg"):
        epsg_cd = int(crs.split(":")[-1])
    else:
        epsg_cd = -1

    attitudes = []

    for ndx, row in geodataframe.iterrows():

        pt = row['geometry']
        x, y = pt.x, pt.y
        dip_dir, dip_ang = row[dip_dir_fldnm], row[dip_ang_fldnm]
        attitudes.append((Point(x, y, epsg_cd=epsg_cd), Plane(dip_dir, dip_ang)))

    return attitudes

