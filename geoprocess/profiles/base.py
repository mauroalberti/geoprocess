from typing import List, Tuple, Optional, Dict

from math import sin, asin, cos, pi, radians, sqrt, isfinite

import numpy as np

import itertools

from pygsf.spatial.vectorial.geometries import *
from pygsf.spatial.rasters.geoarray import GeoArray
from pygsf.spatial.projections.geodetic import *
from pygsf.geology.orientations import *


def profile_parameters(profile: Line) -> Tuple[List[float], List[float], List[float]]:
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


class GeoProfilesSet(object):
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


class LinearProfile(object):
    """
    Class storing a segment profile.
    It is contained within a vertical plane, assuming a Cartesian x-y-z frame.
    """

    def __init__(self, start_pt: Point, end_pt: Point, densify_distance: float):
        """
        Instantiates a 2D linear profile object.
        It is represented by two 2D points and by a densify distance.

        :param start_pt: the profile start point.
        :type start_pt: Point.
        :param end_pt: the profile end point.
        :type end_pt: Point.
        :param densify_distance: the distance with which to densify the segment profile.
        :type densify_distance: float or integer.
        """

        if not isinstance(start_pt, Point):
            raise Exception("Input start point must be a Point instance")

        if not isinstance(end_pt, Point):
            raise Exception("Input end point must be a Point instance")

        if start_pt.crs() != end_pt.crs():
            raise Exception("Both points must have same CRS")

        if start_pt.dist2DWith(end_pt) == 0.0:
            raise Exception("Input segment length cannot be zero")

        if not isinstance(densify_distance, (float, int)):
            raise Exception("Input densify distance must be float or integer")

        if not isfinite(densify_distance):
            raise Exception("Input densify distance must be finite")

        if densify_distance <= 0.0:
            raise Exception("Input densify distance must be positive")

        epsg_cd = start_pt.epsg()

        self._start_pt = Point(x=start_pt.x, y=start_pt.y, epsg_cd=epsg_cd)
        self._end_pt = Point(x=end_pt.x, y=end_pt.y, epsg_cd=epsg_cd)
        self._crs = Crs(epsg_cd)
        self._densify_dist = densify_distance

    def start_pt(self) -> Point:
        """
        Returns a copy of the segment start 2D point.

        :return: start point copy.
        :rtype: Point.
        """

        return self._start_pt.clone()

    def end_pt(self) -> Point:
        """
        Returns a copy of the segment end 2D point.

        :return: end point copy.
        :rtype: Point.
        """

        return self._end_pt.clone()

    def crs(self) -> Crs:
        """
        Returns the CRS of the profile.

        :return: the CRS of the profile.
        :rtype: Crs.
        """

        return Crs(self._crs.epsg())

    def epsg(self) -> float:
        """
        Returns the EPSG code of the profile.

        :return: the EPSG code of the profile.
        :rtype: float.
        """

        return self.crs().epsg()

    def segment(self) -> Segment:
        """
        Returns the horizontal segment representing the profile.

        :return: segment representing the profile.
        :rtype: Segment.
        """

        return Segment(start_pt=self._start_pt, end_pt=self._end_pt)

    def densified_steps(self) -> List[float]:
        """
        Returns a list of the incremental steps (2D distances) along the profile.

        :return: the list of incremental steps.
        :rtype: List[float].
        """

        return self.segment().densify2d_asSteps(self._densify_dist)

    def num_pts(self) -> int:
        """
        Returns the number of steps making up the profile.

        :return: number of steps making up the profile.
        :rtype: int.
        """

        return len(self.densified_points())

    def densified_points(self) -> List[Point]:
        """
        Returns the list of densified 2D points.

        :return: list of densified points.
        :rtype: List[Point].
        """

        return self.segment().densify2d_asPts(densify_distance=self._densify_dist)

    def vertical_plane(self) -> CPlane:
        """
        Returns the vertical plane of the segment, as a Cartesian plane.

        :return: the vertical plane of the segment, as a Cartesian plane.
        :rtype: CPlane.
        """

        return self.segment().vertical_plane()

    def get_z_values(self, grid: GeoArray) -> Optional[List[float]]:
        """

        :param dem:
        :return:
        """

        if self.crs() != grid.crs():
            return None

        return [grid.interpolate_bilinear(pt_2d.x, pt_2d.y) for pt_2d in self.densified_points()]


class TopoProfile(object):
    """
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

    def profile_s(self) -> List[float]:
        """
        Returns the incremental 2D lengths of a profile.

        :return: the incremental 2D lengths.
        :rtype: list of float values.
        """

        return list(itertools.accumulate(self.horiz_dist_values))

    def profile_length(self) -> float:
        """
        Returns the length of the profile.

        :return: length of profile.
        :rtype: float.
        """

        return self._line.length_2d()

    def profile_length_3d(self) -> float:
        """
        Returns the 3D length of the profile.

        :return: 3D length of profile.
        :rtype: float.
        """

        return self._line.length_3d()

    def elevations(self) -> List[float]:
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

    def slopes(self) -> List[Optional[float]]:
        """
        Returns the slopes of a topographic profile.

        :return: slopes.
        :rtype: list of slope values.
        """

        return self._line.slopes()

    def abs_slopes(self) -> List[Optional[float]]:
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


class GeoProfile(object):
    """
    Class representing the topographic and geological elements
    embodying a single geological profile.
    """

    def __init__(self):
        """

        """

        self.topoprofiles       = []  # instance of TopoProfiles
        self.geoplane_attitudes = []
        self.geosurfaces        = []
        self.geosurfaces_ids    = []
        self.lineaments         = []
        self.outcrops           = []

    """
    def sources_names(self):
 
        return [topoprofile.source_name for topoprofile in self.topoprofiles]
    """

    def set_topo_profiles(self, topo_profiles: List[TopoProfile]):
        """

        :param topo_profiles: the topographic profiles.
        :type topo_profiles: List of TopoProfile instances.
        :return:
        """

        self.topoprofiles = topo_profiles

    def add_intersections_pts(self, intersection_list):
        """

        :param intersection_list:
        :return:
        """

        self.lineaments += intersection_list

    def add_intersections_lines(self, formation_list, intersection_line3d_list, intersection_polygon_s_list2):
        """

        :param formation_list:
        :param intersection_line3d_list:
        :param intersection_polygon_s_list2:
        :return:
        """

        self.outcrops = zip(formation_list, intersection_line3d_list, intersection_polygon_s_list2)

    def profiles_svals(self) -> List[List[float]]:
        """
        Returns the list of the s values for the profiles.

        :return: list of the s values.
        :rtype
        """

        return [topoprofile.profile_s() for topoprofile in self.topoprofiles]

    def profiles_zs(self) -> List[float]:
        """
        Returns the elevations of the profiles.

        :return: the elevations.
        :rtype: list of float values.
        """

        return [topoprofile.elevations() for topoprofile in self.topoprofiles]

    def profiles_lengths_2d(self) -> List[float]:
        """
        Returns the 2D lengths of the profiles.

        :return: the 2D profiles lengths.
        :rtype: list of float values.
        """

        return [topoprofile.profile_length for topoprofile in self.topoprofiles]

    def profiles_lengths_3d(self) -> List[float]:
        """
        Returns the 3D lengths of the profiles.

        :return: the 3D profiles lengths.
        :rtype: list of float values.
        """

        return [topoprofile.profile_length_3d() for topoprofile in self.topoprofiles]

    def minmax_length_2d(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns the maximum 2D length of profiles.

        :return: the minimum and maximum profiles lengths.
        :rtype: a pair of optional float values.
        """

        lengths = self.profiles_lengths_2d()

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

        lengths = self.profiles_lengths_2d()

        if lengths:
            return max(lengths)
        else:
            return None

    def topoprofiles_params(self):
        """

        :return:
        """

        topo_lines = [topo_profile.line for topo_profile in self.topoprofiles]
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

        return [topoprofile.elev_stats() for topoprofile in self.topoprofiles]

    def slopes(self) -> List[List[Optional[float]]]:
        """
        Returns a list of the slopes of the topographic profiles in the geoprofile.

        :return: list of the slopes of the topographic profiles.
        :rtype: list of list of topographic slopes.
        """

        return [topoprofile.slopes() for topoprofile in self.topoprofiles]

    def abs_slopes(self) -> List[List[Optional[float]]]:
        """
        Returns a list of the absolute slopes of the topographic profiles in the geoprofile.

        :return: list of the absolute slopes of the topographic profiles.
        :rtype: list of list of topographic absolute slopes.
        """

        return [topoprofile.abs_slopes() for topoprofile in self.topoprofiles]

    def slopes_stats(self) -> List[Dict]:
        """
        Returns the directional slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.slopes_stats() for topoprofile in self.topoprofiles]

    def absslopes_stats(self) -> List[Dict]:
        """
        Returns the absolute slopes statistics
        for each profile.

        :return: the list of the profiles statistics.
        :rtype: List of dictionaries.
        """

        return [topoprofile.absslopes_stats() for topoprofile in self.topoprofiles]

    def min_z_plane_attitudes(self):
        """

        :return:
        """

        # TODO:  manage case for possible nan p_z values
        return min([plane_attitude.pt_3d.p_z for plane_attitude_set in self.geoplane_attitudes for plane_attitude in
                    plane_attitude_set if 0.0 <= plane_attitude.sign_hor_dist <= self.max_s()])

    def max_z_plane_attitudes(self):
        """

        :return:
        """

        # TODO:  manage case for possible nan p_z values
        return max([plane_attitude.pt_3d.p_z for plane_attitude_set in self.geoplane_attitudes for plane_attitude in
                    plane_attitude_set if 0.0 <= plane_attitude.sign_hor_dist <= self.max_s()])

    def min_z_curves(self):
        """

        :return:
        """

        return min([pt_2d.p_y for multiline_2d_list in self.geosurfaces for multiline_2d in multiline_2d_list for line_2d in
                    multiline_2d.lines for pt_2d in line_2d.pts if 0.0 <= pt_2d.p_x <= self.max_s()])

    def max_z_curves(self):
        """

        :return:
        """

        return max([pt_2d.p_y for multiline_2d_list in self.geosurfaces for multiline_2d in multiline_2d_list for line_2d in
                    multiline_2d.lines for pt_2d in line_2d.pts if 0.0 <= pt_2d.p_x <= self.max_s()])

    def mins_z_topo(self) -> List[float]:
        """
        Returns a list of elevation minimums in the topographic profiles.

        :return: the elevation minimums.

        """

        return [topo_profile.elev_stats["min"] for topo_profile in self.topoprofiles]

    def maxs_z_topo(self) -> List[float]:
        """
        Returns a list of elevation maximums in the topographic profiles.

        :return:
        """

        return [topo_profile.elev_stats["max"] for topo_profile in self.topoprofiles]

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

        if len(self.geoplane_attitudes) > 0:
            min_z = min([min_z, self.min_z_plane_attitudes()])

        if len(self.geosurfaces) > 0:
            min_z = min([min_z, self.min_z_curves()])

        return min_z

    def max_z(self):
        """

        :return:
        """

        max_z = self.max_z_topo()

        if len(self.geoplane_attitudes) > 0:
            max_z = max([max_z, self.max_z_plane_attitudes()])

        if len(self.geosurfaces) > 0:
            max_z = max([max_z, self.max_z_curves()])

        return max_z

    def add_plane_attitudes(self, plane_attitudes):
        """

        :param plane_attitudes:
        :return:
        """

        self.geoplane_attitudes.append(plane_attitudes)

    def add_curves(self, lMultilines, lIds):
        """

        :param lMultilines:
        :param lIds:
        :return:
        """

        self.geosurfaces.append(lMultilines)
        self.geosurfaces_ids.append(lIds)


class PlaneAttitude(object):

    def __init__(self, rec_id, source_point_3d, source_geol_plane, point_3d, slope_rad, dwnwrd_sense, sign_hor_dist):
        """

        :param rec_id:
        :param source_point_3d:
        :param source_geol_plane:
        :param point_3d:
        :param slope_rad:
        :param dwnwrd_sense:
        :param sign_hor_dist:
        """

        self.id = rec_id
        self.src_pt_3d = source_point_3d
        self.src_geol_plane = source_geol_plane
        self.pt_3d = point_3d
        self.slope_rad = slope_rad
        self.dwnwrd_sense = dwnwrd_sense
        self.sign_hor_dist = sign_hor_dist

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

def calculate_distance_with_sign(projected_point, section_init_pt, section_vector):
    """

    :param projected_point:
    :param section_init_pt:
    :param section_vector:
    :return:
    """

    assert projected_point.z != np.nan
    assert projected_point.z is not None

    projected_vector = Segment(section_init_pt, projected_point).vector()
    cos_alpha = section_vector.cos_angle(projected_vector)

    return projected_vector.len_3d * cos_alpha


def get_intersection_slope(intersection_versor_3d, section_vector):
    """

    :param intersection_versor_3d:
    :param section_vector:
    :return:
    """

    slope_radians = abs(radians(intersection_versor_3d.slope))
    scalar_product_for_downward_sense = section_vector.sp(intersection_versor_3d.downward)
    if scalar_product_for_downward_sense > 0.0:
        intersection_downward_sense = "right"
    elif scalar_product_for_downward_sense == 0.0:
        intersection_downward_sense = "vertical"
    else:
        intersection_downward_sense = "left"

    return slope_radians, intersection_downward_sense


def calculate_intersection_versor(section_cartes_plane, structural_cartes_plane):
    """

    :param section_cartes_plane:
    :param structural_cartes_plane:
    :return:
    """

    return section_cartes_plane.inters_versor(structural_cartes_plane)


def calculate_nearest_intersection(intersection_versor_3d, section_cartes_plane, structural_cartes_plane,
                                   structural_pt):
    """

    :param intersection_versor_3d:
    :param section_cartes_plane:
    :param structural_cartes_plane:
    :param structural_pt:
    :return:
    """

    dummy_inters_point = section_cartes_plane.inters_point(structural_cartes_plane)
    dummy_structural_vector = Segment(dummy_inters_point, structural_pt).vector()
    dummy_distance = dummy_structural_vector.sp(intersection_versor_3d)
    offset_vector = intersection_versor_3d.scale(dummy_distance)

    return Point(dummy_inters_point.x + offset_vector.x,
                 dummy_inters_point.y + offset_vector.y,
                 dummy_inters_point.z + offset_vector.z)


def calculate_axis_intersection(map_axis, section_cartes_plane, structural_pt):
    """

    :param map_axis:
    :param section_cartes_plane:
    :param structural_pt:
    :return:
    """

    axis_versor = map_axis.as_vect().versor
    l, m, n = axis_versor.x, axis_versor.y, axis_versor.z
    axis_param_line = ParamLine3D(structural_pt, l, m, n)
    return axis_param_line.intersect_cartes_plane(section_cartes_plane)


def map_measure_to_section(geological_pt: Point, geological_plane: Plane, topo_profile: TopoProfile, map_axis=None):
    """

    :param structural_rec:
    :param section_data:
    :param map_axis:
    :return:
    """

    # extract source data

    section_init_pt = topo_profile.start_pt()

    """
    section_cartes_plane, section_vector = section_data['init_pt'], section_data['cartes_plane'], \
                                                            section_data['vector']

    # transform geological plane attitude into Cartesian plane

    geological_cplane = geological_plane.toCPlane(geological_pt)

    # intersection versor

    intersection_versor = calculate_intersection_versor(section_cartes_plane, geological_cplane)

    # calculate slope of geological plane onto section plane

    slope_radians, intersection_downward_sense = get_intersection_slope(intersection_versor, section_vector)

    # intersection point

    if map_axis is None:
        intersection_point_3d = calculate_nearest_intersection(intersection_versor, section_cartes_plane,
                                                               geological_cplane, geological_pt)
    else:
        intersection_point_3d = calculate_axis_intersection(map_axis, section_cartes_plane, geological_pt)

    # horizontal spat_distance between projected structural point and profile start

    signed_distance_from_section_start = calculate_distance_with_sign(intersection_point_3d, section_init_pt,
                                                                      section_vector)

    # solution for current structural point

    return PlaneAttitude(
        structural_pt_id,
        geological_pt,
        geological_plane,
        intersection_point_3d,
        slope_radians,
        intersection_downward_sense,
        signed_distance_from_section_start
    )
    """


def map_struct_pts_on_section(structural_data, section_data, mapping_method):
    """
    defines:
        - 2D x-y location in section
        - plane-plane segment intersection

    :param structural_data:
    :param section_data:
    :param mapping_method:
    :return:
    """

    if mapping_method['method'] == 'nearest':
        return [map_measure_to_section(structural_rec, section_data) for structural_rec in structural_data]

    if mapping_method['method'] == 'common axis':
        map_axis = Axis(mapping_method['trend'], mapping_method['plunge'])
        return [map_measure_to_section(structural_rec, section_data, map_axis) for structural_rec in structural_data]

    if mapping_method['method'] == 'individual axes':
        assert len(mapping_method['individual_axes_values']) == len(structural_data)
        result = []
        for structural_rec, (trend, plunge) in zip(structural_data, mapping_method['individual_axes_values']):
            try:
                map_axis = Axis(trend, plunge)
                result.append(map_measure_to_section(structural_rec, section_data, map_axis))
            except:
                continue
        return result


def extract_multiline2d_list(structural_line_layer, project_crs):
    """

    :param structural_line_layer:
    :param project_crs:
    :return:
    """

    line_orig_crs_geoms_attrs = line_geoms_attrs(structural_line_layer)

    line_orig_geom_list3 = [geom_data[0] for geom_data in line_orig_crs_geoms_attrs]
    line_orig_crs_MultiLine2D_list = [xytuple_l2_to_MultiLine(xy_list2) for xy_list2 in line_orig_geom_list3]
    line_orig_crs_clean_MultiLine2D_list = [multiline_2d.remove_coincident_points() for multiline_2d in
                                            line_orig_crs_MultiLine2D_list]

    # get CRS information
    structural_line_layer_crs = structural_line_layer.crs()

    # project input line layer to project CRS
    line_proj_crs_MultiLine2D_list = [multiline_project(multiline2d, structural_line_layer_crs, project_crs) for
                                          multiline2d in line_orig_crs_clean_MultiLine2D_list]

    return line_proj_crs_MultiLine2D_list


def calculate_profile_lines_intersection(multilines2d_list, id_list, profile_line2d):
    """

    :param multilines2d_list:
    :param id_list:
    :param profile_line2d:
    :return:
    """

    profile_segment2d_list = profile_line2d.as_segments()

    profile_segment2d = profile_segment2d_list[0]

    intersection_list = []
    for ndx, multiline2d in enumerate(multilines2d_list):
        if id_list is None:
            multiline_id = ''
        else:
            multiline_id = id_list[ndx]
        for line2d in multiline2d.lines:
            for line_segment2d in line2d.as_segments():
                try:
                    intersection_point2d = profile_segment2d.intersection_2d_pt(line_segment2d)
                except ZeroDivisionError:
                    continue
                if intersection_point2d is None:
                    continue
                if line_segment2d.contains_2d_pt(intersection_point2d) and \
                   profile_segment2d.contains_2d_pt(intersection_point2d):
                    intersection_list.append([intersection_point2d, multiline_id])

    return intersection_list


def intersection_distances_by_profile_start_list(profile_line, intersections):
    """

    :param profile_line:
    :param intersections:
    :return:
    """

    # convert the profile line from a CartesianLine2DT to a CartesianSegment2DT
    profile_segment2d_list = profile_line.as_segments()
    # debug
    assert len(profile_segment2d_list) == 1
    profile_segment2d = profile_segment2d_list[0]

    # determine distances for each point in intersection list
    # creating a list of float values
    distance_from_profile_start_list = []
    for intersection in intersections:
        distance_from_profile_start_list.append(profile_segment2d.start_pt.dist2DWith(intersection[0]))

    return distance_from_profile_start_list


def define_plot_structural_segment(structural_attitude, profile_length, vertical_exaggeration, segment_scale_factor=70.0):
    """

    :param structural_attitude:
    :param profile_length:
    :param vertical_exaggeration:
    :param segment_scale_factor:
    :return:
    """

    ve = float(vertical_exaggeration)
    intersection_point = structural_attitude.pt_3d
    z0 = intersection_point.z

    h_dist = structural_attitude.sign_hor_dist
    slope_rad = structural_attitude.slope_rad
    intersection_downward_sense = structural_attitude.dwnwrd_sense
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
