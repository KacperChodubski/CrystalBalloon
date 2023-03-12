import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature

class ViewMap:
    def __init__(self):
        self.real_points = []
        self.pred_points = []
        self.ax = plt.axes(projection=ccrs.Miller())
        self.ax.set_extent([-13, 45, 30, 70])
        self.ax.stock_img()
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.BORDERS)
        self.ax.add_feature(cfeature.COASTLINE)

    def add_point(self, lat, lon, real: bool):
        if real:
            self.real_points.append((lat, lon))
            for i in range(len(self.real_points) - 1):

                to_lon = self.real_points[i+1][1]
                to_lat = self.real_points[i+1][0]
                ff_lon = self.real_points[i][1]
                ff_lat = self.real_points[i][0]

                self.ax.plot([to_lon, ff_lon], [to_lat, ff_lat], c='b', lw=1,
                transform=ccrs.PlateCarree())

        else:
            self.pred_points.append((lat, lon))
            for i in range(len(self.pred_points) - 1):

                to_lon = self.pred_points[i+1][1]
                to_lat = self.pred_points[i+1][0]
                ff_lon = self.pred_points[i][1]
                ff_lat = self.pred_points[i][0]

                self.ax.plot([to_lon, ff_lon], [to_lat, ff_lat], c='r', lw=1,
                transform=ccrs.PlateCarree())

    def show_map(self):
        plt.show()


if __name__ == '__main__':
    viewMap = ViewMap()
    viewMap.show_map()