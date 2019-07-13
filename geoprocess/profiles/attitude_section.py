
from typing import Optional


from pygsf.spatial.vectorial.geometries import Point, Segment, CPlane
from pygsf.geology.orientations import Plane


def nearest_attitude_projection(
        section_cplane: CPlane,
        attitude_plane: Plane,
        attitude_pt: Point) -> Optional[Point]:
    """
    Calculates the nearest projection of a given attitude on a vertical plane.

    :param section_cplane: vertical plane.
    :type section_cplane: pygsf.spatial.vectorial.geometries.CPlane.
    :param attitude_plane: geological attitude.
    :type attitude_plane: pygsf.orientations.orientations.Plane
    :param attitude_pt: point of the geological attitude.
    :type attitude_pt: pygsf.spatial.vectorial.geometries.Point.
    :return: the nearest projected point on the vertical section.
    :rtype: pygsf.spatial.vectorial.geometries.Point.
    """

    if section_cplane.crs() != attitude_pt.crs():
        return None

    attitude_cplane = attitude_plane.toCPlane(attitude_pt)
    intersection_versor = section_cplane.intersVersor(attitude_cplane)
    dummy_inters_pt = section_cplane.intersPoint(attitude_cplane)
    dummy_structural_vect = Segment(dummy_inters_pt, attitude_pt).vector()
    dummy_distance = dummy_structural_vect.vDot(intersection_versor)
    offset_vector = intersection_versor.scale(dummy_distance)

    return Point(
        x=dummy_inters_pt.x + offset_vector.x,
        y=dummy_inters_pt.y + offset_vector.y,
        z=dummy_inters_pt.z + offset_vector.z,
        epsg_cd=section_cplane.epsg())


