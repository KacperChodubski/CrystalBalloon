from tkinter.tix import Tree
from model.model import PredictionModel
from data.ecmwf_data_collector import ECMWF_data_collector
from utils.view import ViewMap
import datetime
import torch
import os
import utils.utils as ut
import csv


class Predictor:
    def __init__(self, predictor_config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare the models (up and down) - load the weights and put the model into evaluation mode
        self.model_up = PredictionModel().to(self.device)
        training_state_up = torch.load(os.path.join(predictor_config["model_binaries_path"], predictor_config["model_up_name"]))
        state_dict_up= training_state_up["state_dict"]
        self.model_up.load_state_dict(state_dict_up, strict=True)
        self.model_up.eval()

        self.model_down = PredictionModel().to(self.device)
        training_state_down = torch.load(os.path.join(predictor_config["model_binaries_path"], predictor_config["model_down_name"]))
        state_dict_down = training_state_down["state_dict"]
        self.model_down.load_state_dict(state_dict_down, strict=True)
        self.model_down.eval()

        self.view = ViewMap()
        self.ecmwf = ECMWF_data_collector()
    
    def predict(self, lat, lon, alt, alt0, mass, time: datetime.datetime, burst_altitude: float, n_of_predictions = None):
        with torch.no_grad():
            lat_pred = lat
            lon_pred = lon
            alt_pred = alt
            time_pred = time
            delta_time = datetime.timedelta(seconds=60)
            predictions_n = 0

            burst_alt = burst_altitude
            cur_model = self.model_up

            while  (not n_of_predictions or predictions_n < n_of_predictions) and alt_pred >= 0:
                print('prediction: ', predictions_n)
                predictions_n += 1

                if alt_pred >= burst_alt:
                    cur_model = self.model_down
                
                pressure = ut.calculate_pressure(self.ecmwf, lat_pred, lon_pred, alt_pred, alt0, time_pred)
                temp, wind_u, wind_v = self.ecmwf.get_data(lat_pred, lon_pred, pressure, time_pred)

                input_for_pred = [pressure, mass, temp, wind_u, wind_v, delta_time.total_seconds()]
                input_for_pred = ut.prepare_data(input_for_pred, self.device)

                prediciton = cur_model(input_for_pred)

                self.view.add_point(lat_pred, lon_pred, False)

                lat_pred += prediciton['lat'].item() / 100
                lon_pred += prediciton['lon'].item() / 100
                alt_pred += prediciton['alt'].item() * 1000
                time_pred += delta_time
                print(f'lat:{lat_pred}, lon:{lon_pred}, alt:{alt_pred}')
    
    def add_flight_to_map (self, path):
        with open(path, newline='') as flight_file:
            reader = csv.reader(flight_file, delimiter=',')
            for row in reader:
                lat = row[0]
                lon = row[1]
                self.view.add_point(float(lat), float(lon), True)


if __name__ == '__main__':

    # Setting parameters of flight

    predictions_limit = 40 # limit of made precitions
    lat, lon = 48.2455,11.54762
    alt = 272 # metes
    alt0 = 50 # meters
    mass = 4 # kg (dont change that)
    date = '2023-03-26 10:46:47'

    """
    Selecting type of balloon from:
            h100
            h300
            h350
            h500
            h600
            h800
            h1000
            h1200
            h1600
            h2000
            h3000
    """
    balloon_mass = 'h1200'
    payload = 3500 # grams
    ascent_rate = 5 # m/s

    burst_altitude = ut.calculate_burst_altitude(balloon_mass, payload, ascent_rate)
    burst_altitude = 21750

    model_binaries_path = os.path.join(os.path.dirname(__file__), 'trained_models', 'binaries')

    # Define path to trained models for upward and downward movement
    predictor_config = {
        'model_binaries_path': model_binaries_path,
        'model_up_name': 'model_up.pth',
        'model_down_name': 'model_down.pth',
    }

    date = datetime.datetime.fromisoformat(date)

    predictor = Predictor(predictor_config)
    predictor.predict(lat, lon, alt, alt0, mass, date, burst_altitude)
    predictor.view.show_map()