

def topoprofiles_from_dems(
    canvas: QgsMapCanvas,
    source_profile_line: Line,
    sample_distance: float,
    selected_dems: List,
    selected_dem_parameters: List,
    invert_profile: bool) -> List[TopoProfile]:
    """
    Calculates the topographic profiles from the provided DEMs.

    :param canvas: the QGIS project canvas.
    :type canvas: QgsMapCanvas.
    :param source_profile_line: the source profile.
    :type source_profile_line: Line.
    :param sample_distance: the sampling distance.
    :type sample_distance: float.
    :param selected_dems: the source DEMs.
    :type selected_dems: list of DEMs.
    :param selected_dem_parameters: the selected DEMs parameters.
    :type selected_dem_parameters: list of DEMs parameters.
    :param invert_profile: whether the profile is inverted.
    :type invert_profile: bool.
    :return: TopoProfiles.
    :rtype: list of TopoProfile instance(s).
    """

    # get project CRS information

    project_crs = get_project_crs(canvas)

    if invert_profile:
        line = source_profile_line.reverse_direction()
    else:
        line = source_profile_line

    resampled_line = line.densify_2d_line(sample_distance)  # line resampled by sample distance

    # calculate 3D profiles from DEMs

    crs_authid = get_project_crs_authid(canvas)

    curr_solutions = []
    for dem, dem_params in zip(selected_dems, selected_dem_parameters):

        topo_line = topoline_from_dem(
            resampled_line,
            project_crs,
            dem,
            dem_params)

        curr_solutions.append(TopoProfile(
            crs=crs_authid,
            source_type=DEM_LINE_SOURCE,
            source_name=dem.name(),
            line=topo_line,
            inverted=invert_profile))

    return curr_solutions


def topoprofiles_from_gpxsource(source_gpx_path: str, invert_profile: bool) -> List[TopoProfile]:
    """

    :param source_gpx_path: the gpx file path.
    :type source_gpx_path: basestring.
    :param invert_profile: whether the profile is inverted or no.
    :type invert_profile: bool.
    :return: the topogreaphic profiles.
    :rtype: list storing a single Topoprofile instance.
    """

    doc = xml.dom.minidom.parse(source_gpx_path)

    # define track name
    try:
        trkname = doc.getElementsByTagName('trk')[0].getElementsByTagName('name')[0].firstChild.data
    except:
        trkname = ''

    # get raw track point values (lat, lon, elev, time)
    track_raw_data = []
    for trk_node in doc.getElementsByTagName('trk'):
        for trksegment in trk_node.getElementsByTagName('trkseg'):
            for tkr_pt in trksegment.getElementsByTagName('trkpt'):
                track_raw_data.append(
                    (tkr_pt.getAttribute("lat"),
                     tkr_pt.getAttribute("lon"),
                     tkr_pt.getElementsByTagName("ele")[0].childNodes[0].data,
                     tkr_pt.getElementsByTagName("time")[0].childNodes[0].data))

    # reverse profile orientation if requested
    if invert_profile:
        track_data = track_raw_data[::-1]
    else:
        track_data = track_raw_data

    # create list of Point elements
    track_points = []
    for val in track_data:
        track_points = Point(*val)
        track_points.append(track_points)

    # check for the presence of track points
    if len(track_points) == 0:
        raise GPXIOException("No track point found in this file")

    topo_line = Line(pts=track_points, crs=epsg_4326_str)

    return [TopoProfile(
        crs=epsg_4326_str,
        source_type=GPX_FILE_SOURCE,
        source_name=trkname,
        line=topo_line,
        inverted=invert_profile)]


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


def calculate_projected_3d_pts(canvas, struct_pts, structural_pts_crs, demObj):
    """

    :param canvas:
    :param struct_pts:
    :param structural_pts_crs:
    :param demObj:
    :return:
    """

    demCrs = demObj.params.crs

    # check if on-the-fly-projection is set on
    project_crs = get_project_crs(canvas)

    # set points in the project crs
    if structural_pts_crs != project_crs:
        struct_pts_in_prj_crs = calculate_pts_in_projection(struct_pts, structural_pts_crs, project_crs)
    else:
        struct_pts_in_prj_crs = copy.deepcopy(struct_pts)

        # project the source points from point layer crs to DEM crs
    # if the two crs are different
    if structural_pts_crs != demCrs:
        struct_pts_in_dem_crs = calculate_pts_in_projection(struct_pts, structural_pts_crs, demCrs)
    else:
        struct_pts_in_dem_crs = copy.deepcopy(struct_pts)

    # - 3D structural points, with x, y, and z extracted from the current DEM
    struct_pts_z = get_zs_from_dem(struct_pts_in_dem_crs, demObj)

    assert len(struct_pts_in_prj_crs) == len(struct_pts_z)

    return [Point(pt.x, pt.y, z) for (pt, z) in zip(struct_pts_in_prj_crs, struct_pts_z)]


