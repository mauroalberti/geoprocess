
from typing import List
from operator import attrgetter

import numbers

from math import acos

from pygsf.geology.orientations import *
from pygsf.spatial.rasters.geoarray import *

from ..types.utils import check_type
from ..geology.base import GeorefAttitude
from .chains import *
from .sets import *


class LinearProfiler:
    """
    Class storing a linear (straight) profile.
    It is contained within a vertical plane, assuming a Cartesian x-y-z frame.
    """

    def __init__(self,
            start_pt: Point,
            end_pt: Point,
            densify_distance: float):
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

        if not isinstance(densify_distance, numbers.Real):
            raise Exception("Input densify distance must be a real number")

        if not isfinite(densify_distance):
            raise Exception("Input densify distance must be finite")

        if densify_distance <= 0.0:
            raise Exception("Input densify distance must be positive")

        epsg_cd = start_pt.epsg()

        self._start_pt = Point(x=start_pt.x, y=start_pt.y, epsg_cd=epsg_cd)
        self._end_pt = Point(x=end_pt.x, y=end_pt.y, epsg_cd=epsg_cd)
        self._crs = Crs(epsg_cd)
        self._densify_dist = float(densify_distance)

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

    def densify_dist(self) -> float:
        """
        Returns the densify distance of the profiler.

        :return: the densify distance of the profiler.
        :rtype: float.
        """

        return self._densify_dist

    def __repr__(self):
        """
        Representation of a profile instance.

        :return: the textual representation of the instance.
        :rtype: str.
        """

        return "LinearProfiler(\n\tstart_pt = {},\n\tend_pt = {},\n\tdensify_distance = {})".format(
            self.start_pt(),
            self.end_pt(),
            self.densify_dist()
        )

    def crs(self) -> Crs:
        """
        Returns the CRS of the profile.

        :return: the CRS of the profile.
        :rtype: Crs.
        """

        return Crs(self._crs.epsg())

    def epsg(self) -> int:
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

    def length(self) -> float:
        """
        Returns the length of the profiler section.

        :return: length of the profiler section.
        :rtype: float.
        """

        return self.segment().length_3d()

    def vector(self) -> Vect:
        """
        Returns the horizontal vector representing the profile.

        :return: vector representing the profile.
        :rtype: Vect.
        """

        return self.segment().vector()

    def densified_steps(self) -> array:
        """
        Returns an array made up by the incremental steps (2D distances) along the profile.

        :return: array storing incremental steps values.
        :rtype: array.
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

    def normal_versor(self) -> Vect:
        """
        Returns the perpendicular (horizontal) versor to the profiles (vertical) plane.

        :return: the perpendicular (horizontal) versor to the profiles (vertical) plane.
        :rtype: Vect.
        """

        return self.vertical_plane().normVersor()

    def point_in_profile(self, pt: Point) -> bool:
        """
        Checks whether a point lies in the profiler plane.

        :param pt: the point to check.
        :type pt: Point.
        :return: whether the point lie in the profiler plane.
        :rtype: bool.
        :raise; Exception.
        """

        return self.vertical_plane().isPointInPlane(pt)

    def point_distance(self, pt: Point) -> float:
        """
        Calcultes the point distance from the profiler plane.

        :param pt: the point to check.
        :type pt: Point.
        :return: the point distance from the profiler plane.
        :rtype: float.
        :raise; Exception.
        """

        return self.vertical_plane().pointDistance(pt)

    def sample_grid(
            self,
            grid: GeoArray) -> array:
        """
        Sample grid values along the profiler points.

        :param grid: the input grid
        :type grid: GeoArray.
        :return: array storing the z values sampled from the grid,
        :rtype: array.
        :raise: Exception
        """

        if not isinstance(grid, GeoArray):
            raise Exception("Input grid must be a GeoArray but is {}".format(type(grid)))

        if self.crs() != grid.crs():
            raise Exception("Input grid EPSG code must be {} but is {}".format(self.epsg(), grid.epsg()))

        return array('d', [grid.interpolate_bilinear(pt_2d.x, pt_2d.y) for pt_2d in self.densified_points()])

    def profile_grid(
            self,
            geoarray: GeoArray) -> ScalarProfile:
        """
        Create profile from one geoarray.

        :param geoarray: the source geoarray.
        :type geoarray: GeoArray.
        :return: the profile of the scalar variable stored in the geoarray.
        :rtype: ScalarProfile.
        :raise: Exception.
        """

        check_type(geoarray, "GeoArray", GeoArray)

        return ScalarProfile(
            s_array=self.densified_steps(),
            z_array=self.sample_grid(geoarray))


    def profile_grids(self,
            *grids: Tuple[GeoArray]) -> Optional[ScalarProfiles]:
        """
        Create profiles of one or more grids.

        :param grids: a set of grids, one or more.
        :type grids: tuple of GeoArray instances.
        :return:
        :rtype:
        """

        for grid in grids:
            if not isinstance(grid, GeoArray):
                return None

        grid_profiles = ScalarProfiles(
            s_array=self.densified_steps())

        for grid in grids:

            grid_profiles.add_zs(
                z_array=self.sample_grid(grid))

        return grid_profiles

    def point_signed_s(
            self,
            pt: Point) -> float:
        """
        Calculates the point signed distance from the profiles start.
        The projected point must already lay in the profile vertical plane, otherwise an exception is raised.

        The implementation assumes (and verifies) that the point lies in the profile vertical plane.
        Given that case, it calculates the signed distance from the section start point,
        by using the triangle law of sines.

        :param pt: the point on the section.
        :type pt: Point.
        :return: the signed distance on the profile.
        :rtype: float.
        :raise: Exception.
        """

        if not isinstance(pt, Point):
            raise Exception("Projected point should be Point but is {}".format(type(pt)))

        if self.crs() != pt.crs():
            raise Exception("Projected point should have {} EPSG but has {}".format(self.epsg(), pt.epsg()))

        if not self.point_in_profile(pt):
            raise Exception("Projected point should lie in the profile plane but there is a distance of {} units".format(self.point_distance(pt)))

        projected_vector = Segment(self.start_pt(), pt).vector()
        cos_alpha = self.vector().angleCos(projected_vector)

        return projected_vector.len3D * cos_alpha

    def get_intersection_slope(self,
                intersection_vector: Vect) -> Tuple[float, str]:
        """
        Calculates the slope (in radians) and the downward sense ('left', 'right' or 'vertical')
        for a profile-laying vector.

        :param intersection_vector: the profile-plane lying vector.
        :type intersection_vector: Vect,
        :return: the slope (in radians) and the downward sense.
        :rtype: Tuple[float, str].
        :raise: Exception.
        """

        if not isinstance(intersection_vector, Vect):
            raise Exception("Input argument should be Vect but is {}".format(type(intersection_vector)))

        angle = degrees(acos(self.normal_versor().angleCos(intersection_vector)))
        if abs(90.0 - angle) > 1.0e-4:
            raise Exception("Input argument should lay in the profile plane")

        slope_radians = abs(radians(intersection_vector.slope_degr()))

        scalar_product_for_downward_sense = self.vector().vDot(intersection_vector.downward())
        if scalar_product_for_downward_sense > 0.0:
            intersection_downward_sense = "right"
        elif scalar_product_for_downward_sense == 0.0:
            intersection_downward_sense = "vertical"
        else:
            intersection_downward_sense = "left"

        return slope_radians, intersection_downward_sense

    def calculate_axis_intersection(self,
            map_axis: Axis,
            structural_pt: Point) -> Optional[Point]:
        """
        Calculates the optional intersection point between an axis passing through a point
        and the profiler plane.

        :param map_axis: the projection axis.
        :type map_axis: Axis.
        :param structural_pt: the point through which the axis passes.
        :type structural_pt: Point.
        :return: the optional intersection point.
        :type: Optional[Point].
        :raise: Exception.
        """

        if not isinstance(map_axis, Axis):
            raise Exception("Map axis should be Axis but is {}".format(type(map_axis)))

        if not isinstance(structural_pt, Point):
            raise Exception("Structural point should be Point but is {}".format(type(structural_pt)))

        if self.crs() != structural_pt.crs():
            raise Exception("Structural point should have {} EPSG but has {}".format(self.epsg(), structural_pt.epsg()))

        axis_versor = map_axis.asDirect().asVersor()

        l, m, n = axis_versor.x, axis_versor.y, axis_versor.z

        axis_param_line = ParamLine3D(structural_pt, l, m, n)

        return axis_param_line.intersect_cartes_plane(self.vertical_plane())

    def calculate_intersection_versor(
            self,
            attitude_plane: Plane,
            attitude_pt: Point) -> Optional[Vect]:
        """
        Calculate the intersection versor between the plane profiler and
        a geological plane with location defined by a Point.

        :param attitude_plane:
        :type attitude_plane: Plane,
        :param attitude_pt: the attitude point.
        :type attitude_pt: Point.
        :return:
        """

        if not isinstance(attitude_plane, Plane):
            raise Exception("AttitudePrjct plane should be Plane but is {}".format(type(attitude_plane)))

        if not isinstance(attitude_pt, Point):
            raise Exception("AttitudePrjct point should be Point but is {}".format(type(attitude_pt)))

        if self.crs() != attitude_pt.crs():
            raise Exception("AttitudePrjct point should has EPSG {} but has {}".format(self.epsg(), attitude_pt.epsg()))

        putative_inters_versor = self.vertical_plane().intersVersor(attitude_plane.toCPlane(attitude_pt))

        if not putative_inters_versor.isValid:
            return None

        return putative_inters_versor

    def nearest_attitude_projection(
            self,
            georef_attitude: GeorefAttitude) -> Point:
        """
        Calculates the nearest projection of a given attitude on a vertical plane.

        :param georef_attitude: geological attitude.
        :type georef_attitude: GeorefAttitude
        :return: the nearest projected point on the vertical section.
        :rtype: pygsf.spatial.vectorial.geometries.Point.
        :raise: Exception.
        """

        if not isinstance(georef_attitude, GeorefAttitude):
            raise Exception("georef_attitude point should be GeorefAttitude but is {}".format(type(georef_attitude)))

        if self.crs() != georef_attitude.posit.crs():
            raise Exception("AttitudePrjct point should has EPSG {} but has {}".format(self.epsg(), georef_attitude.posit.epsg()))

        attitude_cplane = georef_attitude.attitude.toCPlane(georef_attitude.posit)
        intersection_versor = self.vertical_plane().intersVersor(attitude_cplane)
        dummy_inters_pt = self.vertical_plane().intersPoint(attitude_cplane)
        dummy_structural_vect = Segment(dummy_inters_pt, georef_attitude.posit).vector()
        dummy_distance = dummy_structural_vect.vDot(intersection_versor)
        offset_vector = intersection_versor.scale(dummy_distance)

        projected_pt = Point(
            x=dummy_inters_pt.x + offset_vector.x,
            y=dummy_inters_pt.y + offset_vector.y,
            z=dummy_inters_pt.z + offset_vector.z,
            epsg_cd=self.epsg())

        return projected_pt

    def map_attitude_to_section(
            self,
            georef_attitude: GeorefAttitude,
            map_axis: Optional[Axis] = None) -> Optional[AttitudePrjct]:
        """
        Project a georeferenced attitude to the section.

        :param georef_attitude: the georeferenced attitude.
        :type georef_attitude: GeorefAttitude.
        :param map_axis: the map axis.
        :type map_axis: Optional[Axis].
        :return: the optional planar attitude on the profiler vertical plane.
        :rtype: Optional[PlanarAttitude].
        """

        if not isinstance(georef_attitude, GeorefAttitude):
            raise Exception("Georef attitude should be GeorefAttitude but is {}".format(type(georef_attitude)))

        if self.crs() != georef_attitude.posit.crs():
            raise Exception("AttitudePrjct point should has EPSG {} but has {}".format(self.epsg(), georef_attitude.posit.epsg()))

        if map_axis:
            if not isinstance(map_axis, Axis):
                raise Exception("Map axis should be Axis but is {}".format(type(map_axis)))

        # intersection versor

        intersection_versor = self.calculate_intersection_versor(
            attitude_plane=georef_attitude.attitude,
            attitude_pt=georef_attitude.posit
        )

        # calculate slope of geological plane onto section plane

        slope_radians, intersection_downward_sense = self.get_intersection_slope(intersection_versor)

        # intersection point

        if map_axis is None:
            intersection_point_3d = self.nearest_attitude_projection(
                georef_attitude=georef_attitude)
        else:
            intersection_point_3d = self.calculate_axis_intersection(
                map_axis=map_axis,
                structural_pt=georef_attitude.posit)

        if not intersection_point_3d:
            return None

        # distance along projection vector

        dist = georef_attitude.posit.dist3DWith(intersection_point_3d)

        # horizontal spat_distance between projected structural point and profile start

        signed_distance_from_section_start = self.point_signed_s(intersection_point_3d)

        # solution for current structural point

        return AttitudePrjct(
            id=georef_attitude.id,
            s=signed_distance_from_section_start,
            z=intersection_point_3d.z,
            slope_degr=degrees(slope_radians),
            down_sense=intersection_downward_sense,
            dist=dist
        )

    def map_georef_attitudes_to_section(
        self,
        structural_data: List[GeorefAttitude],
        mapping_method: dict,
        height_source: Optional[GeoArray] = None) -> List[Optional[AttitudePrjct]]:
        """
        Projects a set of georeferenced attitudes onto the section profile,
        optionally extracting point heights from a grid.

        defines:
            - 2D x-y position in section
            - plane-plane segment intersection

        :param structural_data: the set of georeferenced attitudes to plot on the section.
        :type structural_data: List[GeorefAttitude]
        :param mapping_method: the method to map the attitudes to the section.
        ;type mapping_method; Dict.
        :param height_source: the attitudes elevation source. Default is None.
        :type height: Optional[GeoArray].
        :return: sorted list of ProfileAttitude values.
        :rtype: List[Optional[ProfileAttitude]].
        :raise: Exception.
        """

        if height_source:

            if not isinstance(height_source, GeoArray):
                raise Exception("Height source should be GeoArray but is {}".format(type(height_source)))

            attitudes_3d = []
            for georef_attitude in structural_data:
                pt3d = height_source.interpolate_bilinear_point(
                    pt=georef_attitude.posit)
                if pt3d:
                    attitudes_3d.append(GeorefAttitude(
                        georef_attitude.id,
                        pt3d,
                        georef_attitude.attitude))

        else:

            attitudes_3d = structural_data

        if mapping_method['method'] == 'nearest':
            results = [self.map_attitude_to_section(georef_att) for georef_att in attitudes_3d]
        elif mapping_method['method'] == 'common axis':
            map_axis = Axis(mapping_method['trend'], mapping_method['plunge'])
            results = [self.map_attitude_to_section(georef_att, map_axis) for georef_att in attitudes_3d]
        elif mapping_method['method'] == 'individual axes':
            if len(mapping_method['individual_axes_values']) != len(attitudes_3d):
                raise Exception(
                    "Individual axes values are {} but attitudes are {}".format(
                        len(mapping_method['individual_axes_values']),
                        len(attitudes_3d)
                    )
                )

            results = []
            for georef_att, (trend, plunge) in zip(attitudes_3d, mapping_method['individual_axes_values']):
                try:
                    map_axis = Axis(trend, plunge)
                    results.append(self.map_attitude_to_section(georef_att, map_axis))
                except:
                    continue

        return sorted(results, key=attrgetter('s'))


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

    # determine distances for each position in intersection list
    # creating a list of float values
    distance_from_profile_start_list = []
    for intersection in intersections:
        distance_from_profile_start_list.append(profile_segment2d.start_pt.dist2DWith(intersection[0]))

    return distance_from_profile_start_list


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


def profile_polygon_intersection(profile_qgsgeometry, polygon_layer, inters_polygon_classifaction_field_ndx):
    """

    :param profile_qgsgeometry:
    :param polygon_layer:
    :param inters_polygon_classifaction_field_ndx:
    :return:
    """

    intersection_polyline_polygon_crs_list = []

    if polygon_layer.selectedFeatureCount() > 0:
        features = polygon_layer.selectedFeatures()
    else:
        features = polygon_layer.getFeatures()

    for polygon_feature in features:
        # retrieve every (selected) feature with its geometry and attributes

        # fetch geometry
        poly_geom = polygon_feature.geometry()

        intersection_qgsgeometry = poly_geom.intersection(profile_qgsgeometry)

        try:
            if intersection_qgsgeometry.isEmpty():
                continue
        except:
            try:
                if intersection_qgsgeometry.isGeosEmpty():
                    continue
            except:
                return False, "Missing function for checking empty geometries.\nPlease upgrade QGIS"

        if inters_polygon_classifaction_field_ndx >= 0:
            attrs = polygon_feature.attributes()
            polygon_classification = attrs[inters_polygon_classifaction_field_ndx]
        else:
            polygon_classification = None

        if intersection_qgsgeometry.isMultipart():
            lines = intersection_qgsgeometry.asMultiPolyline()
        else:
            lines = [intersection_qgsgeometry.asPolyline()]

        for line in lines:
            intersection_polyline_polygon_crs_list.append([polygon_classification, line])

    return True, intersection_polyline_polygon_crs_list