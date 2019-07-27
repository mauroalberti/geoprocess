# -*- coding: utf-8 -*-


import unittest


from pygsf.spatial.vectorial.geometries import *

from geoprocess.profiles.profilers import LinearProfiler

src_profile_shapefile_pth = "/home/mauro/Documents/projects/geoprocess/example_data/mt_alpi/profile.shp"
attitudes_shape = "/home/mauro/Documents/projects/geoprocess/example_data/mt_alpi/attitudes.shp"


class TestProfiles(unittest.TestCase):

    def setUp(self):

        pass

    def test_nearest_attitude_projection(self):

        """
        profiles = read_linestring_geometries(src_profile_shapefile_pth)
        line = profiles.extract_line()

        profiler = LinearProfiler(
            start_pt=line.start_pt(),
            end_pt=line.end_pt(),
            densify_distance=5)  # meters
        """

        profiler = LinearProfiler(
            start_pt=Point(581981.1880, 4442999.2144, 0.0000, 0.0000, 32633),
            end_pt=Point(586203.5723, 4440126.1905, 0.0000, 0.0000, 32633),
            densify_distance=5.0)

        attitude = GeorefAttitude(
            location=Point(583531.3753, 4441614.4005, 0.0000, 0.0000, 32633),
            attitude=Plane(090.00, +18.00))

        """
        attitudes = geopandas.read_file(attitudes_shape)

        attitudes = extract_georeferenced_attitudes(
            geodataframe=attitudes,
            dip_dir_fldnm="dip_dir",
            dip_ang_fldnm="dip_ang")
        """

        projected_pt = profiler.map_attitude_to_section(
            georef_attitude=attitude,
            map_axis=None)

        """
        mapping_method = {}
        mapping_method['method'] = 'nearest'
        att_projs = profiler.map_georef_attitudes_to_section(
            structural_data=attitudes,
            mapping_method=mapping_method,
            height_source=geoarray)
        """

        assert areClose(pl.angle(CPlane(1, 0, 1, 0, epsg_cd=2000)), 0.)

    def tearDown(self):

        pass


if __name__ == '__main__':
    unittest.main()
