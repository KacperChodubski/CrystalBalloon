import ecmwf_data_collector as edc
import sondehub_data_collector as sdc
import csv
import datetime
import math
import os

sondehub_DC = sdc.Sondehub_data_collector()
ecmwf_DC = edc.ECMWF_data_collector()
    

def create_dataset(start, finish):
    
    sondehub_data_frame = sondehub_DC.get_dataFrame()

    finish = min(finish, sondehub_data_frame.shape[0]-1)

    dir_path = os.path.dirname(__file__)

    dataset_up_path = os.path.join(dir_path, "balloon/datasets_up.csv")
    dataset_down_path = os.path.join(dir_path, "balloon/datasets_down.csv")

    if start == 0:
        f_up = open(dataset_up_path, 'w')
        f_down = open(dataset_down_path, 'w')
    else: 
        f_up = open(dataset_up_path, 'a')
        f_down = open(dataset_down_path, 'a')
    writer_up = csv.writer(f_up, delimiter=',')
    writer_down = csv.writer(f_down, delimiter=',')

    for i in range(start, finish, 1):

        percent_done = round(((i/sondehub_data_frame.shape[0]) * 100))

        
        print(f' {percent_done}% done','#' * math.floor(percent_done/5), end="\r")
        sdf = sondehub_data_frame.iloc[[i]]
        sdf_next = sondehub_data_frame.iloc[[i+1]]
        if (sdf['serial'].item() != sdf_next['serial'].item()):
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

        sdf_date = datetime.datetime.fromisoformat(sdf['datetime'].values.item())
        temp, wind_u, wind_v = ecmwf_DC.get_data(lat, lon, pressure, datetime = sdf_date)

        temp = float(temp)
        wind_u = float(wind_u)
        wind_v = float(wind_v)
        lat_dif = float(lat_dif)
        lon_dif = float(lon_dif)
        alt_dif = float(alt_dif)
        deltatime = (dt_next - dt).total_seconds()

        row = [pressure, mass, temp, wind_u, wind_v, deltatime, lat_dif * 100, lon_dif * 100, alt_dif / 1000]
        if (alt_dif < -50):
            writer_down.writerow(row)
        else:
            writer_up.writerow(row)

    





if __name__ == '__main__':
    # First line in sondehub_datas.csv file
    start = 6487
    # Last line in sondehub_datas.csv file
    finish = 6489
    create_dataset(start=start, finish=finish)
        
