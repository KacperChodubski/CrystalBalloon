import math

from numpy import dtype
from model.model import PredictionModel
from data.ecmwf_data_collector import ECMWF_data_collector
from utils.view import ViewMap
from data.data_module import BalloonDataset
import datetime
import torch
import time as ttime
import os
import utils.utils as ut


class Predictor:
    def __init__(self):
        self.ecmwf = ECMWF_data_collector()

        cur_path = os.path.dirname(__file__)
        path_up = os.path.join(cur_path, 'data/balloon/datasets_up.csv')
        path_down = os.path.join(cur_path, 'data/balloon/datasets_down.csv')

        dataset_up = BalloonDataset(path_up)
        dataset_down = BalloonDataset(path_down)
        self.model_up = PredictionModel(dataset=dataset_up)
        self.model_down = PredictionModel(dataset=dataset_down)
        num_epochs = 30
        self.model_up.train(num_epochs)
        self.model_down.train(num_epochs)

        self.view = ViewMap()
    
    def predict(self, lat, lon, alt, alt0, mass, time: datetime.datetime, burst_altitude: float):
        with torch.no_grad():
            lat_pred = lat
            lon_pred = lon
            alt_pred = alt
            time_pred = time
            delta_time = datetime.timedelta(seconds=60)
            predictions_n = 0

            burst_alt = burst_altitude
            cur_model = self.model_up

            while predictions_n < 180 and alt_pred >= 0:
                print('prediction: ', predictions_n)
                predictions_n += 1

                if alt_pred >= burst_alt:
                    cur_model = self.model_down
                
                pressure = ut.calculate_pressure(self.ecmwf, lat_pred, lon_pred, alt_pred, alt0, time_pred)
                temp, wind_u, wind_v = self.ecmwf.get_data(lat_pred, lon_pred, pressure, time_pred)

                input_for_pred = [pressure, mass, temp, wind_u, wind_v, delta_time.total_seconds()]
                input_for_pred = torch.tensor(input_for_pred, dtype=torch.float32)
                input_for_pred = (input_for_pred - cur_model.mean) / cur_model.std

                prediciton = cur_model(input_for_pred)

                self.view.add_point(lat_pred, lon_pred, False)

                lat_pred += prediciton['lat'].item() / 100
                lon_pred += prediciton['lon'].item() / 100
                alt_pred += prediciton['alt'].item() * 1000
                time_pred += delta_time
                print(f'lat:{lat_pred}, lon:{lon_pred}, alt:{alt_pred}')

            
            self.view.show_map()


if __name__ == '__main__':

    lat, lon = 53.5965, 19.5513
    alt = 272
    alt0 = 50
    mass = 4
    date = '2023-03-20 20:00:00'

    balloon_mass = 'h1600'
    payload = 3500
    ascent_rate = 5

    burst_altitude = ut.calculate_burst_altitude(balloon_mass, payload, ascent_rate)
    #burst_altitude = 36567

    predictor = Predictor()
    predictor.predict(lat, lon, alt, alt0, mass, datetime.datetime.fromisoformat(date), burst_altitude)