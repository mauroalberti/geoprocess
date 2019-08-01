
from matplotlib import gridspec
import matplotlib.pyplot as plt

from ..widgets.mpl_widget import MplWidget

from .sets import *


z_padding = 0.2


class GeoProfile:

    """
    Class representing the topographic and geological elements
    embodying a single geological profile.
    """

    def __init__(self,
         topo_profile: Optional[TopographicProfile] = None,
         attitudes: Optional[Attitudes] = None,
         lines_intersections: Optional[LinesIntersections] = None,
         polygons_intersections: Optional[PolygonsIntersections] = None
    ):
        """

        """

        if topo_profile:
            check_type(topo_profile, "Topographic profile", TopographicProfile)

        if attitudes:
            check_type(attitudes, "Attitudes", Attitudes)

        if lines_intersections:
            check_type(lines_intersections, "Line intersections", LinesIntersections)

        if polygons_intersections:
            check_type(polygons_intersections, "Polygon intersections", PolygonsIntersections)

        self._topo_profile = topo_profile
        self._attitudes = attitudes
        self._lines_intersections = lines_intersections
        self._polygons_intersections = polygons_intersections

    @property
    def topo_profile(self):
        """

        :return:
        """

        return self._topo_profile

    @topo_profile.setter
    def topo_profile(self,
        scalars_inters: TopographicProfile):
        """

        :param scalars_inters: the scalar values profiles.
        :type scalars_inters: TopographicProfile.
        :return:

        """

        check_type(scalars_inters, "Topographic profile", TopographicProfile)
        self._topo_profile = scalars_inters

    def clear_topo_profile(self):
        """

        :return:
        """

        self._topo_profile = None

    def plot_topo_profile(self,
          aspect: numbers.Real = 1,
          width: numbers.Real = 18.5,
          height: numbers.Real = 10.5,
          color="blue"):
        """
        Plot a set of profiles with Matplotlib.

        :param aspect: the plot aspect.
        :type aspect: numbers.Real.
        :param width: the plot width, in inches. # TOCHECK IF ALWAYS INCHES
        :type width: numbers.Real.
        :param height: the plot height in inches.  # TOCHECK IF ALWAYS INCHES
        :type color: the color.
        :param color: str.
        :type height: numbers.Real.
        :return: the figure.
        :rtype:
        """

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

        if self._topo_profile:
            ax.plot(
                self._topo_profile.s(),
                self._topo_profile.z(),
                color=color
            )

        self.fig = fig

    @property
    def attitudes(self):
        """

        :return:
        """

        return self._attitudes

    @attitudes.setter
    def attitudes(self,
                  prj_attitudes: Attitudes):
        """
        Set the projected _attitudes content.

        :param prj_attitudes: projected _attitudes.
        :type prj_attitudes: Attitudes.
        :return:
        """

        check_type(prj_attitudes, "Projected _attitudes", List)
        for el in prj_attitudes:
            check_type(el, "Projected attitude", Attitude)

        self._attitudes = prj_attitudes

    def clear_attitudes(self):
        """
        Clear projected _attitudes content.

        :return:
        """

        self._attitudes = None

    def plot_attitudes(self, color="red"):
        """

        :return:
        """

        self.fig = self._attitudes.plot(
            self.fig,
            self.length_2d(),
            color=color
        )

    def plot(self,
             topo_profile=True,
             attitudes=True,
             line_intersections=True,
             polygon_intersections=True,
             topo_profile_color="blue",
             attitudes_color="red",
             line_intersections_color="orange",
             aspect: numbers.Real = 1,
             width: numbers.Real = 18.5,
             height: numbers.Real = 10.5,
             **params
             ):
        """

        :param topo_profile:
        :param attitudes:
        :param line_intersections:
        :param polygon_intersections:
        :param line_projections:
        :return:
        """

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

        if 'plot_z_min' in params and 'plot_z_max' in params:
            plot_z_min = params['plot_z_min']
            plot_z_max = params['plot_z_max']
        else:

            z_range = self.z_max() - self.z_min()
            plot_z_min = self.z_min() - z_padding * z_range
            plot_z_max = self.z_max() + z_padding * z_range

        if topo_profile and self._topo_profile:

            ax.set_ylim([plot_z_min, plot_z_max])
            ax.plot(
                self._topo_profile.s(),
                self._topo_profile.z(),
                color=topo_profile_color
            )

        if attitudes and self._attitudes:

            self._attitudes.plot(
                fig,
                self.length_2d(),
                color=attitudes_color
            )

        if line_intersections and self._lines_intersections:

            self._lines_intersections.plot(
                fig,
                self.length_2d(),
                color=line_intersections_color
            )

        if polygon_intersections and self._polygons_intersections:

            self._polygons_intersections.plot(
                fig,
                self.length_2d()
            )

    def s_min(self):
        """

        :return:
        """

        return self.topo_profile.s_min()

    def s_max(self):
        """

        :return:
        """

        return self.topo_profile.s_max()

    def z_min(self):
        """

        :return:
        """

        return self.topo_profile.z_min()

    def z_max(self):
        """

        :return:
        """

        return self.topo_profile.z_max()

    '''
    def add_intersections_pts(self, intersection_list):
        """

        :param intersection_list:
        :return:
        """

        self._lines_intersections += intersection_list

    def add_intersections_lines(self, formation_list, intersection_line3d_list, intersection_polygon_s_list2):
        """

        :param formation_list:
        :param intersection_line3d_list:
        :param intersection_polygon_s_list2:
        :return:
        """

        self._polygons_intersections = zip(formation_list, intersection_line3d_list, intersection_polygon_s_list2)
    '''

    def length_2d(self) -> numbers.Real:
        """
        Returns the 2D length of the profile.

        :return: the 2D profile length.
        :rtype: numbers.Real.
        """

        return self._topo_profile.profile_length()


class GeoProfileSet():
    """
    Represents a set of Geoprofile elements,
    stored as a list
    """

    def __init__(self,
        topo_profiles_set: Optional[TopographicProfileSet] = None,
        attitudes_set: Optional[AttitudesSet] = None,
        lines_intersections_set: Optional[LinesIntersectionsSet] = None,
        polygons_intersections_set: Optional[PolygonsIntersectionsSet] = None
        ):

        if topo_profiles_set:
            check_type(topo_profiles_set, "Toppographic profiles set", TopographicProfileSet)

        if attitudes_set:
            check_type(attitudes_set, "Attitudes set", AttitudesSet)

        if lines_intersections_set:
            check_type(lines_intersections_set, "Lines intersections set", LinesIntersectionsSet)

        if polygons_intersections_set:
            check_type(polygons_intersections_set, "Polygons_intersections set", PolygonsIntersectionsSet)

        self._topo_profiles_set = topo_profiles_set
        self._attitudes_set = attitudes_set
        self._lines_intersections_set = lines_intersections_set
        self._polygons_intersections_set = polygons_intersections_set

    def parameters(self) -> List:
        """
        Returns all the attributes of the class.

        :return:
        """

        return [
            self._topo_profiles_set,
            self._attitudes_set,
            self._lines_intersections_set,
            self._polygons_intersections_set
        ]

    @property
    def topo_profiles_set(self):
        """

        :return:
        """

        return self._topo_profiles_set

    @topo_profiles_set.setter
    def topo_profiles_set(self,
        topo_profiles_set: TopographicProfileSet):
        """

        :param scalars_inters: the scalar values profiles.
        :type scalars_inters: TopographicProfile.
        :return:

        """

        check_type(topo_profiles_set, "Topographic profiles set", TopographicProfileSet)
        self._topo_profiles_set = topo_profiles_set

    @property
    def attitudes_set(self):
        """

        :return:
        """

        return self._attitudes_set

    @attitudes_set.setter
    def attitudes_set(self,
        attitudes_set: AttitudesSet):
        """

        :param attitudes_set: the attitudes set.
        :type attitudes_set: AttitudesSet.
        :return:

        """

        check_type(attitudes_set, "Attitudes set", AttitudesSet)
        self._attitudes_set = attitudes_set

    @property
    def lines_intersections_set(self):
        """

        :return:
        """

        return self._lines_intersections_set

    @lines_intersections_set.setter
    def lines_intersections_set(self,
                                lines_intersections_set: LinesIntersectionsSet):
        """

        :param lines_intersections_set: the lines intersections set.
        :type lines_intersections_set: LinesIntersectionsSet.
        :return:

        """

        check_type(lines_intersections_set, "Line intersections set", LinesIntersectionsSet)
        self._lines_intersections_set = lines_intersections_set

    @property
    def polygons_intersections_set(self):
        """

        :return:
        """

        return self._polygons_intersections_set

    @polygons_intersections_set.setter
    def polygons_intersections_set(self,
        polygons_intersections_set: PolygonsIntersectionsSet):
        """

        :param polygons_intersections_set: the polygons intersections set.
        :type polygons_intersections_set: PolygonsIntersectionsSet.
        :return:

        """

        check_type(polygons_intersections_set, "Polygons intersections set", PolygonsIntersectionsSet)
        self._polygons_intersections_set = polygons_intersections_set

    def num_profiles(self) -> numbers.Integral:
        """
        Returns the number of profiles in the geoprofile set.

        :return: number of profiles in the geoprofile set.
        :rtype: numbers.Integral.
        """

        return max(map(lambda lst: len(lst) if lst else 0, self.parameters()))

    def extract_geoprofile(self, ndx: numbers.Integral) -> GeoProfile:
        """
        Returns a geoprofile referencing slices of stored data.

        :param ndx: the index of the geoprofile.
        :type ndx: numbers.Integral.
        :return: the extracted Geoprofile or None.
        :rtype: GeoProfile.
        :raise: Exception.
        """

        if ndx not in range(self.num_profiles()):
            raise Exception("Geoprofile set range is in 0-{} but {} got".format(self.num_profiles() - 1, ndx))

        return GeoProfile(
            topo_profile=self.topo_profiles_set[ndx] if self.topo_profiles_set and ndx < len(self.topo_profiles_set) else None,
            attitudes=self.attitudes_set[ndx] if self.attitudes_set and ndx < len(self.attitudes_set) else None,
            lines_intersections=self.lines_intersections_set[ndx] if self.lines_intersections_set and ndx < len(self.lines_intersections_set) else None,
            polygons_intersections=self.polygons_intersections_set[ndx] if self.polygons_intersections_set and ndx < len(self.polygons_intersections_set) else None
        )

    def s_min(self):
        """

        :return:
        """

        return self.topo_profiles_set.s_min()

    def s_max(self):
        """

        :return:
        """

        return self.topo_profiles_set.s_max()

    def z_min(self):
        """

        :return:
        """

        return self.topo_profiles_set.z_min()

    def z_max(self):
        """

        :return:
        """

        return self.topo_profiles_set.z_max()

    def plot(self):
        """


        :return:
        """

        num_subplots = self.num_profiles()

        z_range = self.z_max() - self.z_min()
        plot_z_min = self.z_min() - z_range * z_padding
        plot_z_max = self.z_max() + z_range * z_padding

        for ndx in range(num_subplots):

            geoprofile = self.extract_geoprofile(ndx)
            geoprofile.plot(
                plot_z_min=plot_z_min,
                plot_z_max=plot_z_max
            )

    '''
    def z_min(self):
        """

        :return:
        """

        z_min = self.min_z_topo()

        if len(self._attitudes) > 0:
            z_min = min([z_min, self.min_z_plane_attitudes()])

        if len(self.traces_projections) > 0:
            z_min = min([z_min, self.min_z_curves()])

        return z_min

    def z_max(self):
        """

        :return:
        """

        z_max = self.max_z_topo()

        if len(self._attitudes) > 0:
            z_max = max([z_max, self.max_z_plane_attitudes()])

        if len(self.traces_projections) > 0:
            z_max = max([z_max, self.max_z_curves()])

        return z_max
    '''

    ## inherited - TO CHECK

    def profiles_svals(self) -> List[List[numbers.Real]]:
        """
        Returns the list of the s values for the profiles.

        :return: list of the s values.
        :rtype
        """

        return [topoprofile.profile_s() for topoprofile in self._topo_profile]

    def profiles_zs(self) -> List[numbers.Real]:
        """
        Returns the elevations of the profiles.

        :return: the elevations.
        :rtype: list of numbers.Real values.
        """

        return [topoprofile.elevations() for topoprofile in self._topo_profile]

    def profiles_lengths_3d(self) -> List[numbers.Real]:
        """
        Returns the 3D lengths of the profiles.

        :return: the 3D profiles lengths.
        :rtype: list of numbers.Real values.
        """

        return [topoprofile.profile_length_3d() for topoprofile in self._topo_profile]

    def minmax_length_2d(self) -> Tuple[Optional[numbers.Real], Optional[numbers.Real]]:
        """
        Returns the maximum 2D length of profiles.

        :return: the minimum and maximum profiles lengths.
        :rtype: a pair of optional numbers.Real values.
        """

        lengths = self.length_2d()

        if lengths:
            return min(lengths), max(lengths)
        else:
            return None, None

    def max_length_2d(self) -> Optional[numbers.Real]:
        """
        Returns the maximum 2D length of profiles.

        :return: the maximum profiles lengths.
        :rtype: an optional numbers.Real value.
        """

        lengths = self.length_2d()

        if lengths:
            return max(lengths)
        else:
            return None


    def add_plane_attitudes(self, plane_attitudes):
        """

        :param plane_attitudes:
        :return:
        """

        self._attitudes.append(plane_attitudes)


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

        # plot geological _attitudes intersections

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