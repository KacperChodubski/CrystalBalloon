import pandas
import ecmwf_data_collector as edc
import sondehub_data_collector as sdc
import pandas as pd
import csv
import datetime
import math

sondehub_DC = sdc.Sondehub_data_collector()
ecmwf_DC = edc.ECMWF_data_collector()

def create_dataset(train = False):

    steps_forward = 300

    if not train:
        sondehub_data_frame = sondehub_DC.get_test_dataFrame()
        dataset_path = './balloon/datasets_testing.csv'
    else:
        sondehub_data_frame = sondehub_DC.get_train_dataFrame()
        dataset_path = './balloon/datasets.csv'


    dataset = []

    f = open(dataset_path, 'w')
    writer = csv.writer(f, delimiter=',')

    for i in range(round((sondehub_data_frame.shape[0]-steps_forward)/ 10)):

        percent_done = round(((i/sondehub_data_frame.shape[0]) * 100))

        
        print(f' {percent_done}% done','#' * math.floor(percent_done/5), end="\r")
        sdf = sondehub_data_frame.iloc[[i]]
        sdf_next = sondehub_data_frame.iloc[[i+steps_forward]]

        try:
            lat_next = float(sdf_next['lat'])
            lon_next = float(sdf_next['lon'])
            alt_next = float(sdf_next['alt'])

            lat = float(sdf['lat'])
            lon = float(sdf['lon'])
            alt = float(sdf['alt'])

            #vel_h = float(sdf['vel_h'])
            #vel_v = float(sdf['vel_v'])
            pressure = float(sdf['pressure'])
            mass = float(sdf['mass'])
        except:
            continue

        lat_dif = (lat_next - lat)*100
        lon_dif = (lon_next - lon)*100
        alt_dif = (alt_next - alt)/1000

        if (alt_dif < -50):
            continue

        sdf_date = datetime.datetime.fromisoformat(sdf['datetime'].values.item())
        temp, wind_u, wind_v = ecmwf_DC.get_data(lat, lon, pressure, datetime = sdf_date)

        #row = [lat, lon, alt, vel_h, vel_v, pressure, mass, temp, wind_u, wind_v]
        row = [lat, lon, alt, pressure, mass, temp, wind_u, wind_v, lat_dif, lon_dif, alt_dif]
        dataset.append(row)
    
    writer.writerows(dataset)
    





if __name__ == '__main__':
    create_dataset(train=True)
    create_dataset(train=False)
        
