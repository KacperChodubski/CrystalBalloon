import datetime
from email import header
from operator import index
from tokenize import String
from wsgiref import headers
from ecmwf.opendata import Client
import pandas as pd
import pygrib
import math
from os import listdir
from os.path import isfile

class ECMWF_data_collector:
    def __init__(self):
        self.client = Client(source="ecmwf")
        self.file_path = './balloon/forecast2023-03-07.grib2'
        self.dir_path = './balloon/'
        self.starting_time = 0
        self.steps = [0, 3, 6, 9, 12, 15, 18, 21]
        self.levels = [1000, 925, 850, 700, 500, 300, 250, 200, 50]
        self.params = ["t", "v", "u"]

    def _round_pressure(self, pressure):
        min_val = 100000
        index = 0
        for i in range(len(self.levels)):
            if abs(self.levels[i] - pressure) < min_val:
                index = i
                min_val = self.levels[i] - pressure
        return self.levels[index]
    
    def _step_of_datetime(self, baloon_date: datetime.datetime):
        min_val = datetime.timedelta.max
        index = 0
        for i in range(len(self.steps)):
            time_of_baloon = datetime.timedelta(hours=baloon_date.hour, minutes=baloon_date.minute)
            time_of_step = datetime.timedelta(hours=self.starting_time+self.steps[i])
            if abs(time_of_step - time_of_baloon) < min_val:
                index = i
                min_val = abs(time_of_step - time_of_baloon)
        return self.steps[index]

    def download_data(self, date: datetime.date):
        self.file_path = './balloon/forecast'+str(date)+'.grib2'
        self.request = {
            'stream':   'oper',
            'date':     str(date),
            'time':     self.starting_time,
            'type':     "fc",
            'step':     self.steps,
            'param':    self.params,
            'levelist': self.levels,
        }
        res = self.client.retrieve(request=self.request, target=self.file_path)
        return self.file_path

    def set_data_file(self, path: String):
        self.file_path = path

    def get_data(self, lat, lon, pressure, datetime: datetime.datetime):
        file_forecast_balloon = self.dir_path+'forecast'+ str(datetime.date())+'.grib2'
        if not isfile((file_forecast_balloon)):
            self.download_data(datetime.date())
        grbs = pygrib.open(file_forecast_balloon)

        pressure = self._round_pressure(pressure)
        step_of_balloon = self._step_of_datetime(datetime)

        index_lat = math.ceil((90 - lat) / 0.4)
        index_lon = math.ceil((lon + 180) / 0.4)

        grbs.seek(0)

        temp: float = None
        wind_u: float = None
        wind_v: float = None

        for grb in grbs:
            if grb.level == pressure and grb.step == step_of_balloon:
                if grb.shortName == "t":
                    temp = grb.values[index_lat][index_lon]
                if grb.shortName == "u":
                    wind_u = grb.values[index_lat][index_lon]
                if grb.shortName == "v":
                    wind_v = grb.values[index_lat][index_lon]

        grbs.close()
        if temp == None or wind_u == None or wind_v == None:
            print('Warning: there isnt temp, wind_u and wind_v in ecmwf data')
        return temp, wind_u, wind_v

if __name__ == '__main__':
    ecmwf = ECMWF_data_collector()
    #ecmwf.download_data('2023-03-07')
    temp, wind_u, wind_v = ecmwf.get_data(54.5189, 18.5319, 13.49, datetime=datetime.datetime.fromisoformat('2023-03-09 07:07:58'))
    print(temp-273)
