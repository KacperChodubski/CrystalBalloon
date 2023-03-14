import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import scipy
import scipy.interpolate 
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


class ViewMap:
    def __init__(self):
        self.real_points_lats = []
        self.real_points_lons = []
        self.pred_points_lats = []
        self.pred_points_lons = []

        request = cimgt.GoogleTiles()

        fig = plt.figure(figsize=(13, 8))
        self.ax = plt.axes(projection=request.crs)
        gl = self.ax.gridlines(draw_labels=True, alpha=0.2)
        gl.top_labels = gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        self.ax.set_extent([-13, 45, 30, 70])
        self.ax.add_image(request, 6, interpolation='spline36')
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.BORDERS)
        self.ax.add_feature(cfeature.COASTLINE)

    def add_point(self, lat, lon, real: bool):
        if real:
            self.real_points_lats.append(lat)
            self.real_points_lons.append(lon)
            for i in range(len(self.real_points_lons) - 1):

                #scipy.interpolate.interp1d(self.real_points_lats, self.real_points_lons, kind='cubic')

                to_lon = self.real_points_lons[i+1]
                to_lat = self.real_points_lats[i+1]
                ff_lon = self.real_points_lons[i]
                ff_lat = self.real_points_lats[i]

                self.ax.plot([to_lon, ff_lon], [to_lat, ff_lat] , c='b', lw=1,
                transform=ccrs.PlateCarree())

        else:
            self.pred_points_lats.append(lat)
            self.pred_points_lons.append(lon)
            for i in range(len(self.pred_points_lats) - 1):

                to_lon = self.pred_points_lons[i+1]
                to_lat = self.pred_points_lats[i+1]
                ff_lon = self.pred_points_lons[i]
                ff_lat = self.pred_points_lats[i]

                self.ax.plot([to_lon, ff_lon], [to_lat, ff_lat], c='r', lw=1,
                transform=ccrs.PlateCarree())

    def show_map(self):
        plt.show()


if __name__ == '__main__':
    viewMap = ViewMap()
    viewMap.add_point(57, 14, True)
    viewMap.add_point(57, 13, True)
    viewMap.add_point(56, 13, True)
    viewMap.show_map()