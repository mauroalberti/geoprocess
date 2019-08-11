
from builtins import str

from typing import Optional, Tuple, List

from math import floor, ceil

import numpy as np

from osgeo import osr

from qgis.core import *
from qgis.gui import *

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from pygsf.spatial.vectorial.geometries import Point, Segment, Line, MultiLine
from pygsf.spatial.exceptions import VectorIOException


def calculate_azimuth_correction(src_pt: Point, crs: QgsCoordinateReferenceSystem) -> numbers.Real:
    """
    Calculates the empirical azimuth correction (angle between y-axis direction and geographic North)
    for a given point.

    :param src_pt: the point for which to calculate the correction.
    :type src_pt: Point.
    :param crs: the considered coordinate reference system.
    :type crs: QgsCoordinateReferenceSystem.
    :return: the azimuth angle.
    :rtype: numbers.Real.
    """

    # Calculates dip direction correction with respect to project CRS y-axis orientation

    srcpt_prjcrs_x = src_pt.x
    srcpt_prjcrs_y = src_pt.y

    srcpt_epsg4326_lon, srcpt_epsg4326_lat = qgs_project_xy(
        x=srcpt_prjcrs_x,
        y=srcpt_prjcrs_y,
        srcCrs=crs)

    north_dummpy_pt_lon = srcpt_epsg4326_lon  # no change
    north_dummpy_pt_lat = srcpt_epsg4326_lat + (1.0 / 1200.0)  # add 3 minute-seconds (approximately 90 meters)

    dummypt_prjcrs_x, dummypt_prjcrs_y = qgs_project_xy(
        x=north_dummpy_pt_lon,
        y=north_dummpy_pt_lat,
        destCrs=crs)

    start_pt = Point(
        srcpt_prjcrs_x,
        srcpt_prjcrs_y)

    end_pt = Point(
        dummypt_prjcrs_x,
        dummypt_prjcrs_y)

    north_vector = Segment(
        start_pt=start_pt,
        end_pt=end_pt).vector()

    azimuth_correction = north_vector.azimuth

    return azimuth_correction


def line_project(line: Line, srcCrs: QgsCoordinateReferenceSystem, destCrs: QgsCoordinateReferenceSystem) -> Line:
    """
    Projects a line from a source to a destination CRS.

    :param line: the original line, to be projected.
    :type line: Line.
    :param srcCrs: the CRS of the original line.
    :type srcCrs: QgsCoordinateReferenceSystem.
    :param destCrs: the final CRS of the line.
    :type destCrs: QgsCoordinateReferenceSystem.
    :return: the projected line.
    :rtype: Line.
    """

    points = []
    for point in line.pts():
        x, y, z = point.toXYZ()
        x, y = qgs_project_xy(
            x=x,
            y=y,
            srcCrs=srcCrs,
            destCrs=destCrs)
        points.append(Point(x, y, z))

    return Line(points)


def multiline_project(multiline: MultiLine, srcCrs: QgsCoordinateReferenceSystem, destCrs: QgsCoordinateReferenceSystem) -> MultiLine:
    """
    Projects a multiline1 from a source to a destination CRS.

    :param multiline: the original multiline1, to be projected.
    :type multiline: MultiLine.
    :param srcCrs: the CRS of the original multiline1.
    :type srcCrs: QgsCoordinateReferenceSystem.
    :param destCrs: the final CRS of the multiline1.
    :type destCrs: QgsCoordinateReferenceSystem.
    :return: the projected multiline1.
    :rtype: MultiLine.
    """

    lines = []
    for line in multiline.lines():
        lines.append(multiline_project(line, srcCrs, destCrs))

    return MultiLine(lines)


def topoline_from_dem(resampled_trace2d: Line, project_crs, dem, dem_params) -> Line:
    """

    :param resampled_trace2d:
    :param project_crs:
    :param dem:
    :param dem_params:
    :return: the Line instance.
    """

    if dem.crs() != project_crs:
        trace2d_in_dem_crs = line_project(resampled_trace2d, project_crs, dem.crs())
    else:
        trace2d_in_dem_crs = resampled_trace2d

    lnProfile = Line()

    for trace_pt2d_dem_crs, trace_pt2d_project_crs in zip(trace2d_in_dem_crs.pts(), resampled_trace2d.pts()):

        fInterpolatedZVal = interpolate_z(dem, dem_params, trace_pt2d_dem_crs)

        pt3dtPoint = Point(
            x=trace_pt2d_project_crs.x,
            y=trace_pt2d_project_crs.y,
            z=fInterpolatedZVal)
        
        lnProfile.add_pt(pt3dtPoint)

    return lnProfile


def calculate_pts_in_projection(pts_in_orig_crs, srcCrs, destCrs):
    """

    :param pts_in_orig_crs:
    :param srcCrs:
    :param destCrs:
    :return:
    """

    pts_in_prj_crs = []
    for pt in pts_in_orig_crs:
        qgs_pt = QgsPointXY(pt.x, pt.y)
        qgs_pt_prj_crs = qgs_project_point(qgs_pt, srcCrs, destCrs)
        pts_in_prj_crs.append(Point(qgs_pt_prj_crs.x(), qgs_pt_prj_crs.y()))
    return pts_in_prj_crs


def intersect_with_dem(demLayer, demParams, project_crs, lIntersPts):
    """

    :param demLayer:
    :param demParams:
    :param project_crs:
    :param lIntersPts:
    :return: a list of Point instances
    """

    # project to Dem CRS
    if demParams.crs != project_crs:
        lQgsPoints = [QgsPointXY(pt.x, pt.y) for pt in lIntersPts]
        lDemCrsIntersQgsPoints = [qgs_project_point(qgsPt, project_crs, demParams.crs) for qgsPt in
                                  lQgsPoints]
        lDemCrsIntersPts = [Point(qgispt.x(), qgispt.y()) for qgispt in lDemCrsIntersQgsPoints]
    else:
        lDemCrsIntersPts = lIntersPts

    # interpolate z values from Dem
    lZVals = [interpolate_z(demLayer, demParams, pt) for pt in lDemCrsIntersPts]

    lXYZVals = [(pt2d.x, pt2d.y, z) for pt2d, z in zip(lIntersPts, lZVals)]

    return [Point(x, y, z) for x, y, z in lXYZVals]


class DEMParams(object):

    def __init__(self, layer, params):
        """

        :param layer:
        :param params:
        """

        self.layer = layer
        self.params = params


def qcolor2rgbmpl(qcolor):

    red = qcolor.red() / 255.0
    green = qcolor.green() / 255.0
    blue = qcolor.blue() / 255.0
    return red, green, blue


def xy_from_canvas(canvas, position):

    mapPos = canvas.getCoordinateTransform().toMapCoordinates(position["x"], position["y"])

    return mapPos.x(), mapPos.y()


def get_prjcrs_as_proj4str(canvas) -> str:

    project_crs = get_project_crs(canvas)
    proj4_str = str(project_crs.toProj4())
    project_crs_osr = osr.SpatialReference()
    project_crs_osr.ImportFromProj4(proj4_str)

    return project_crs_osr


def get_project_crs(canvas: QgsMapCanvas) -> QgsCoordinateReferenceSystem:
    """
    Returns the canvas projection.

    :param canvas: the map canvas.
    :type canvas: QgsMapCanvas.
    :return:the map canvas CRS.
    :rtype: QgsCoordinateReferenceSystem.
    """

    return canvas.mapSettings().destinationCrs()


def get_project_crs_authid(canvas: QgsMapCanvas) -> str:
    """
    Returns the CRS authid (e.g., "EPSG:4326").

    :param canvas:the map canvas.
    :type canvas: QgsMapCanvas.
    :return:the map canvas CRS authid.
    :rtype: basestring.
    """

    return get_project_crs(canvas).authid()


def get_zs_from_dem(struct_pts_2d, demObj):
    # TODO: check if required

    z_list = []
    for point_2d in struct_pts_2d:
        interp_z = interpolate_z(demObj.layer, demObj.params, point_2d)
        z_list.append(interp_z)

    return z_list


def vector_type(layer):
    
    if not layer.type() == QgsMapLayer.VectorLayer:
        raise VectorIOException("Layer is not vector")
    
    if layer.geometryType() == QgsWkbTypes.PointGeometry:
        return "point"
    elif layer.geometryType() == QgsWkbTypes.LineGeometry:
        return "line"        
    elif layer.geometryType() == QgsWkbTypes.PolygonGeometry:
        return "polygon"
    else: 
        raise VectorIOException("Unknown vector type") 
       
    
def loaded_layers():
    
    return list(QgsProject.instance().mapLayers().values())

    
def loaded_vector_layers():
 
    return [layer for layer in loaded_layers() if layer.type() == QgsMapLayer.VectorLayer]
    

def loaded_polygon_layers():

    return [layer for layer in loaded_vector_layers() if layer.geometryType() == QgsWkbTypes.PolygonGeometry]


def loaded_line_layers():        
    
    return [layer for layer in loaded_vector_layers() if layer.geometryType() == QgsWkbTypes.LineGeometry]


def loaded_point_layers():

    return [layer for layer in loaded_vector_layers() if layer.geometryType() == QgsWkbTypes.PointGeometry]
    
 
def loaded_raster_layers():
          
    return [layer for layer in loaded_layers() if layer.type() == QgsMapLayer.RasterLayer]


def loaded_monoband_raster_layers():
          
    return [layer for layer in loaded_raster_layers() if layer.bandCount() == 1]
       

def lyr_attrs(layer, fields: List) -> List:
    """
    Get attributes from layer based on provided field names list.

    :param layer: the source layer for attribute extraction.
    :param fields: a list of field names for attribute extraction.
    :return: list of table values.
    """

    if layer.selectedFeatureCount() > 0:
        features = layer.selectedFeatures()
    else:
        features = layer.getFeatures()

    provider = layer.dataProvider()
    field_indices = [provider.fieldNameIndex(field_name) for field_name in fields]

    # retrieve selected features with relevant attributes

    rec_list = []

    for feature in features:

        attrs = feature.fields().toList()

        # creates feature attribute list

        feat_list = []
        for field_ndx in field_indices:
            feat_list.append(feature.attribute(attrs[field_ndx].name()))

        # add to result list

        rec_list.append(feat_list)

    return rec_list


def ptlyr_geoms_attrs(pt_layer, fields=None) -> Tuple[List[Tuple[numbers.Real, numbers.Real]], List]:
    """
    Extract geometry and attribute informatiion from a point layer.

    :param pt_layer: source layer.
    :param fields: list of field names for information extraction.
    :return: two lists. the first of point coordinates, the second of record values.
    """

    if not fields:
        fields = []

    if pt_layer.selectedFeatureCount() > 0:
        features = pt_layer.selectedFeatures()
    else:
        features = pt_layer.getFeatures()

    provider = pt_layer.dataProvider()
    field_indices = [provider.fieldNameIndex(field_name) for field_name in fields]

    # retrieve selected features with their geometry and relevant attributes
    lGeoms = []
    lAttrs = []

    for feature in features:

        # fetch point geometry

        pt = feature.geometry().asPoint()  # note: it's an x-y point, eventual z is discarded
        lGeoms.append((pt.x(), pt.y()))

        # get attribute values

        attrs = feature.fields().toList()

        rec_vals = []
        for field_ndx in field_indices:
            rec_vals.append(feature.attribute(attrs[field_ndx].name()))

        lAttrs.append(rec_vals)

    return lGeoms, lAttrs


def pt_geoms_attrs(pt_layer, field_list=None) -> List:
    """
    Deprecated: use instead ptlyr_geoms_attrs.

    :param pt_layer: source layer.
    :param field_list: list of field names for information extraction.
    :return: list of values.
    """

    if not field_list:
        field_list = []
    
    if pt_layer.selectedFeatureCount() > 0:
        features = pt_layer.selectedFeatures()
    else:
        features = pt_layer.getFeatures() 
    
    provider = pt_layer.dataProvider()    
    field_indices = [provider.fieldNameIndex(field_name) for field_name in field_list]

    # retrieve selected features with their geometry and relevant attributes
    rec_list = [] 
    for feature in features:
             
        # fetch point geometry
        pt = feature.geometry().asPoint()

        attrs = feature.fields().toList() 

        # creates feature attribute list
        feat_list = [pt.x(), pt.y()]
        for field_ndx in field_indices:
            feat_list.append(str(feature.attribute(attrs[field_ndx].name())))

        # add to result list
        rec_list.append(feat_list)
        
    return rec_list


def line_geoms_attrs(line_layer, field_list=None):

    if not field_list:
        field_list = []

    lines = []
    
    if line_layer.selectedFeatureCount() > 0:
        features = line_layer.selectedFeatures()
    else:
        features = line_layer.getFeatures()

    provider = line_layer.dataProvider() 
    field_indices = [provider.fieldNameIndex(field_name) for field_name in field_list]
                
    for feature in features:
        geom = feature.geometry()
        if geom.isMultipart():
            rec_geom = multipolyline_to_xytuple_list2(geom.asMultiPolyline())
        else:
            rec_geom = [polyline_to_xytuple_list(geom.asPolyline())]
            
        attrs = feature.fields().toList()
        rec_data = [str(feature.attribute(attrs[field_ndx].name())) for field_ndx in field_indices]
            
        lines.append([rec_geom, rec_data])
            
    return lines
           
       
def line_geoms_with_id(line_layer, curr_field_ndx):
        
    lines = []
    progress_ids = [] 
    dummy_progressive = 0 
      
    line_iterator = line_layer.getFeatures()
   
    for feature in line_iterator:
        try:
            progress_ids.append(int(feature[curr_field_ndx]))
        except:
            dummy_progressive += 1
            progress_ids.append(dummy_progressive)
             
        geom = feature.geometry()         
        if geom.isMultipart():
            lines.append(('multiline1', multipolyline_to_xytuple_list2(geom.asMultiPolyline()))) # typedef QVector<QgsPolyline>
            # now is a list of list of (x,y) tuples
        else:           
            lines.append(('line', polyline_to_xytuple_list(geom.asPolyline())))  # typedef QVector<QgsPointXY>
                         
    return lines, progress_ids
              
                   
def polyline_to_xytuple_list(qgsline):
    
    assert len(qgsline) > 0
    return [(qgspoint.x(), qgspoint.y()) for qgspoint in qgsline]


def multipolyline_to_xytuple_list2(qgspolyline):
    
    return [polyline_to_xytuple_list(qgsline) for qgsline in qgspolyline]


def field_values(layer, curr_field_ndx):
    
    values = []
    iterator = layer.getFeatures()
    
    for feature in iterator:
        values.append(feature.attributes()[curr_field_ndx])
            
    return values
    
    
def vect_attrs(layer, field_list):
    
    if layer.selectedFeatureCount() > 0:
        features = layer.selectedFeatures()
    else:
        features = layer.getFeatures()
        
    provider = layer.dataProvider()   
    field_indices = [provider.fieldNameIndex(field_name) for field_name in field_list]

    # retrieve (selected) attributes features
    data_list = [] 
    for feature in features:        
        attrs = feature.fields().toList()     
        data_list.append([feature.attribute(attrs[field_ndx].name()) for field_ndx in field_indices])
        
    return data_list    
    
    
def raster_qgis_params(raster_layer):
    
    name = raster_layer.name()
                  
    rows = raster_layer.height()
    cols = raster_layer.width()
    
    extent = raster_layer.extent()
    
    xMin = extent.xMinimum()
    xMax = extent.xMaximum()        
    yMin = extent.yMinimum()
    yMax = extent.yMaximum()
        
    cellsizeEW = (xMax-xMin) / float(cols)
    cellsizeNS = (yMax-yMin) / float(rows)
    
    #TODO: get real no data value from QGIS
    if raster_layer.dataProvider().sourceHasNoDataValue(1):
        nodatavalue = raster_layer.dataProvider().sourceNoDataValue(1)
    else:
        nodatavalue = np.nan
    
    try:
        crs = raster_layer.crs()
    except:
        crs = None
    
    return name, cellsizeEW, cellsizeNS, rows, cols, xMin, xMax, yMin, yMax, nodatavalue, crs    


def qgs_point(x: numbers.Real, y: numbers.Real) -> QgsPointXY:
    """
    Creates a QgsPointXY instance from x-y coordinates.

    :param x: the x coordinate.
    :type x: numbers.Real.
    :param y: the y coordinate.
    :type y: numbers.Real.
    :return: the QgsPointXY instance.
    :rtype: QgsPointXY instance.
    """
    
    return QgsPointXY(x, y)


def explode_pt(qgs_pt: QgsPointXY) -> Tuple[numbers.Real, numbers.Real]:
    """
    Returns the x and y coordinates of a QgsPointXY.

    :param qgs_pt: a point.
    :type qgs_pt: QgsPointXY instance.
    :return: the x and y pair.
    :rtype: tuple of two numbers.Real values.
    """

    return qgs_pt.x(), qgs_pt.y()


def qgs_project_point(qgsPt: QgsPointXY, srcCrs: QgsCoordinateReferenceSystem = None, destCrs: QgsCoordinateReferenceSystem = None) -> QgsPointXY:
    """
    Project a QGIS point to a new CRS.
    If the source/destination CRS is not provided, it will be set to EPSG 4236 (WGS-84).

    :param qgsPt: the source point.
    :type qgsPt: a QgsPointXY instance.
    :param srcCrs: the source CRS.
    :type srcCrs: QgsCoordinateReferenceSystem.
    :param destCrs: the destination CRS.
    :type destCrs: QgsCoordinateReferenceSystem.
    :return: the projected point.
    :rtype: QgsPointXY instance.
    """

    if not srcCrs:
        srcCrs = QgsCoordinateReferenceSystem(
            4326,
            QgsCoordinateReferenceSystem.EpsgCrsId)

    if not destCrs:
        destCrs = QgsCoordinateReferenceSystem(
            4326,
            QgsCoordinateReferenceSystem.EpsgCrsId)

    coordinate_transform = QgsCoordinateTransform(
        srcCrs,
        destCrs,
        QgsProject.instance())

    prj_pt = coordinate_transform.transform(
        qgsPt)

    return prj_pt


def qgs_project_xy(x: numbers.Real, y: numbers.Real, srcCrs: QgsCoordinateReferenceSystem = None, destCrs:Optional[QgsCoordinateReferenceSystem] = None) -> Tuple[numbers.Real, numbers.Real]:
    """
    Project a pair of x-y coordinates to a new projection.
    If the source/destination CRS is not provided, it will be set to EPSG 4236 (WGS-84).

    :param x: the x coordinate.
    :type x: numbers.Real.
    :param y: the y coordinate.
    :type y: numbers.Real.
    :param srcCrs: the source coordinate.
    :type srcCrs: QgsCoordinateReferenceSystem.
    :param destCrs: the destination coordinate.
    :type destCrs: QgsCoordinateReferenceSystem.
    :return: the projected x-y coordinates.
    :rtype: tuple of two numbers.Real values.
    """

    if not srcCrs:
        srcCrs = QgsCoordinateReferenceSystem(
            4326,
            QgsCoordinateReferenceSystem.EpsgCrsId)

    if not destCrs:
        destCrs = QgsCoordinateReferenceSystem(
            4326,
            QgsCoordinateReferenceSystem.EpsgCrsId)

    coordinate_transform = QgsCoordinateTransform(
        srcCrs,
        destCrs,
        QgsProject.instance())

    qgs_pt = coordinate_transform.transform(
        x,
        y)

    x, y = qgs_pt.x(), qgs_pt.y()

    return x, y


def project_line_2d(srcLine, srcCrs, destCrs):
    
    destLine = Line()
    for pt in srcLine.pts():
        srcPt = QgsPointXY(pt.x, pt.y)
        destPt = qgs_project_point(srcPt, srcCrs, destCrs)
        destLine = destLine.add_pt(Point(destPt.x(), destPt.y()))
        
    return destLine


class QGisRasterParameters(object):

    def __init__(self, name, cellsizeEW, cellsizeNS, rows, cols, xMin, xMax, yMin, yMax, nodatavalue, crs):

        self.name = name
        self.cellsizeEW = cellsizeEW
        self.cellsizeNS = cellsizeNS
        self.rows = rows
        self.cols = cols
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.nodatavalue = nodatavalue
        self.crs = crs

    def point_in_dem_area(self, point):
        """
        Check that a point is intersect or on the boundary of the grid area.
        Assume grid has no rotation.

        :param point: qProf.gsf.geometry.Point
        :return: bool
        """

        if self.xMin <= point.x <= self.xMax and \
                self.yMin <= point.y <= self.yMax:
            return True
        else:
            return False

    def point_in_interpolation_area(self, point):
        """
        Check that a point is intersect or on the boundary of the area defined by
        the extreme cell center values.
        Assume grid has no rotation.

        :param point: qProf.gsf.geometry.Point
        :return: bool
        """

        if self.xMin + self.cellsizeEW / 2.0 <= point.x <= self.xMax - self.cellsizeEW / 2.0 and \
                self.yMin + self.cellsizeNS / 2.0 <= point.y <= self.yMax - self.cellsizeNS / 2.0:
            return True
        else:
            return False

    def geogr2raster(self, point):
        """
        Convert from geographic to raster-based coordinates.
        Assume grid has no rotation.

        :param point: qProf.gsf.geometry.Point
        :return: dict
        """

        x = (point.x - (self.xMin + self.cellsizeEW / 2.0)) / self.cellsizeEW
        y = (point.y - (self.yMin + self.cellsizeNS / 2.0)) / self.cellsizeNS

        return dict(x=x, y=y)

    def raster2geogr(self, array_dict):
        """
        Convert from raster-based to geographic coordinates.
        Assume grid has no rotation.

        :param array_dict: dict
        :return: qProf.gsf.geometry.Point instance
        """

        assert 'x' in array_dict
        assert 'y' in array_dict

        x = self.xMin + (array_dict['x'] + 0.5) * self.cellsizeEW
        y = self.yMin + (array_dict['y'] + 0.5) * self.cellsizeNS

        return Point(x, y)


def get_z(dem_layer, point):

    identification = dem_layer.dataProvider().identify(QgsPointXY(point.x, point.y), QgsRaster.IdentifyFormatValue)
    if not identification.isValid():
        return np.nan
    else:
        try:
            result_map = identification.results()
            return float(result_map[1])
        except:
            return np.nan


def interpolate_bilinear(dem, qrpDemParams, point):
    """
    :param dem: qgis._core.QgsRasterLayer
    :param qrpDemParams: qProf.gis_utils.qgs_tools.QGisRasterParameters
    :param point: qProf.gis_utils.features.Point
    :return: numbers.Real
    """

    dArrayCoords = qrpDemParams.geogr2raster(point)

    floor_x_raster = floor(dArrayCoords["x"])
    ceil_x_raster = ceil(dArrayCoords["x"])
    floor_y_raster = floor(dArrayCoords["y"])
    ceil_y_raster = ceil(dArrayCoords["y"])

    # bottom-left center
    p1 = qrpDemParams.raster2geogr(dict(x=floor_x_raster,
                                        y=floor_y_raster))
    # bottom-right center
    p2 = qrpDemParams.raster2geogr(dict(x=ceil_x_raster,
                                        y=floor_y_raster))
    # top-left center
    p3 = qrpDemParams.raster2geogr(dict(x=floor_x_raster,
                                        y=ceil_y_raster))
    # top-right center
    p4 = qrpDemParams.raster2geogr(dict(x=ceil_x_raster,
                                        y=ceil_y_raster))

    z1 = get_z(dem, p1)
    z2 = get_z(dem, p2)
    z3 = get_z(dem, p3)
    z4 = get_z(dem, p4)

    delta_x = point.x - p1.x
    delta_y = point.y - p1.y

    z_x_a = z1 + (z2 - z1) * delta_x / qrpDemParams.cellsizeEW
    z_x_b = z3 + (z4 - z3) * delta_x / qrpDemParams.cellsizeEW

    return z_x_a + (z_x_b - z_x_a) * delta_y / qrpDemParams.cellsizeNS


def interpolate_z(dem, dem_params, point):
    """
        dem_params: type qProf.gis_utils.qgs_tools.QGisRasterParameters
        point: type qProf.gis_utils.features.Point
    """

    if dem_params.point_in_interpolation_area(point):
        return interpolate_bilinear(dem, dem_params, point)
    elif dem_params.point_in_dem_area(point):
        return get_z(dem, point)
    else:
        return np.nan


"""
Modified from: profiletool, script: tools/ptmaptool.py

#-----------------------------------------------------------
# 
# Profile
# Copyright (C) 2008  Borys Jurgiel
# Copyright (C) 2012  Patrice Verchere
#-----------------------------------------------------------
# 
# licensed under the terms of GNU GPL 2
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, print to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#---------------------------------------------------------------------
"""

class PointMapToolEmitPoint(QgsMapToolEmitPoint):

    def __init__(self, canvas, button):
        
        super(PointMapToolEmitPoint, self).__init__(canvas)
        self.canvas = canvas
        self.cursor = QCursor(Qt.CrossCursor)
        self.button = button


    def setCursor(self, cursor):
        
        self.cursor = QCursor(cursor)
        



class MapDigitizeTool(QgsMapTool):

    moved = pyqtSignal(dict)
    leftClicked = pyqtSignal(dict)
    rightClicked = pyqtSignal(dict)

    def __init__(self, canvas):

        QgsMapTool.__init__(self, canvas)
        self.canvas = canvas
        self.cursor = QCursor(Qt.CrossCursor)

    def canvasMoveEvent(self, event):

        self.moved.emit({'x': event.pos().x(), 'y': event.pos().y()})

    def canvasReleaseEvent(self, event):

        if event.button() == Qt.RightButton:
            self.rightClicked.emit({'x': event.pos().x(), 'y': event.pos().y()})
        elif event.button() == Qt.LeftButton:
            self.leftClicked.emit({'x': event.pos().x(), 'y': event.pos().y()})
        else:
            return

    def canvasDoubleClickEvent(self, event):

        self.doubleClicked.emit({'x': event.pos().x(), 'y': event.pos().y()})

    def activate(self):

        QgsMapTool.activate(self)
        self.canvas.setCursor(self.cursor)

    def deactivate(self):

        QgsMapTool.deactivate(self)

    def isZoomTool(self):

        return False

    def setCursor(self, cursor):

        self.cursor = QCursor(cursor)

