import pandas
from torch import _sparse_compressed_tensor_unsafe
import ecmwf_data_collector as edc
import sondehub_data_collector as sdc
import pandas as pd
import csv
import datetime
import math
import os

sondehub_DC = sdc.Sondehub_data_collector()
ecmwf_DC = edc.ECMWF_data_collector()
    

def create_dataset():

    steps_forward = 1
    start = 1500
    finish = 2000

    
    sondehub_data_frame = sondehub_DC.get_dataFrame()

    finish = min(finish, sondehub_data_frame.shape[0]-steps_forward)

    dir_path = os.path.dirname(__file__)

    dataset_path = os.path.join(dir_path, "balloon/datasets.csv")

    if start == 0:
        f = open(dataset_path, 'w')
    else: 
        f = open(dataset_path, 'a')
    writer = csv.writer(f, delimiter=',')

    for i in range(start, finish, 1):

        percent_done = round(((i/sondehub_data_frame.shape[0]) * 100))

        
        print(f' {percent_done}% done','#' * math.floor(percent_done/5), end="\r")
        sdf = sondehub_data_frame.iloc[[i]]
        sdf_next = sondehub_data_frame.iloc[[i+steps_forward]]
        if (sdf['serial'].values[0] != sdf_next['serial'].values[0]):
            continue

        try:
            lat_next = float(sdf_next['lat'])
            lon_next = float(sdf_next['lon'])
            alt_next = float(sdf_next['alt'])
            dt_next = datetime.datetime.fromisoformat(sdf_next.datetime.item())
            dt = datetime.datetime.fromisoformat(sdf.datetime.item())

            lat = float(sdf['lat'])
            lon = float(sdf['lon'])
            alt = float(sdf['alt'])

            pressure = float(sdf['pressure'])
            mass = float(sdf['mass'])
        except:
            print('Bad data')
            continue

        lat_dif = (lat_next - lat)
        lon_dif = (lon_next - lon)
        alt_dif = (alt_next - alt)

        if (alt_dif < -50):
            continue

        sdf_date = datetime.datetime.fromisoformat(sdf['datetime'].values.item())
        temp, wind_u, wind_v = ecmwf_DC.get_data(lat, lon, pressure, datetime = sdf_date)

        temp = float(temp)
        wind_u = float(wind_u)
        wind_v = float(wind_v)
        lat_dif = float(lat_dif)
        lon_dif = float(lon_dif)
        alt_dif = float(alt_dif)
        deltatime = (dt_next - dt).total_seconds()

        row = [lat, lon, alt, pressure, mass, temp, wind_u, wind_v, deltatime, lat_dif * 100, lon_dif * 100, alt_dif / 1000]
        writer.writerow(row)
    





if __name__ == '__main__':
    create_dataset()
        
