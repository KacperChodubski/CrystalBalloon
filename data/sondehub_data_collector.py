import csv
from distutils.command.build_scripts import first_line_re
from email import header
from math import fabs
from queue import Empty
from random import random, randrange
from tokenize import String
import numpy as np
import pandas as pd
import sondehub
import datetime as date
from dateutil import parser
import os
import random




class Sondehub_data_collector:
    def __init__(self):
        self.header_list = ('serial','datetime','alt','lat', 'lon','vel_h', 'vel_v', 'pressure', 'mass')
        self.cur_dir = os.path.dirname(__file__)
        self.dataFrame: pd.DataFrame = None
        self.max_data_rate = 90
        random.seed()

        
        self.file_path = os.path.join(self.cur_dir, "balloon/sondehub_datas.csv")

    def _convert_datatime(self, sonde_datetime: str):
        substr = sonde_datetime[:10] + '-' + sonde_datetime[11:19]
        datetime = date.datetime.fromisoformat(substr)
        return datetime

    def download_data(self, serial_n: String, datetime: date.date):
        frames = sondehub.download(serial=serial_n, datetime_prefix=datetime.isoformat())
        if (len(frames) > 0):
            self.dataFrame = pd.DataFrame.from_dict(frames, orient='columns')

            self.dataFrame['datetime'] = self.dataFrame['datetime'].apply(self._convert_datatime)

            if (frames[0].get('mass') == None):
                self.dataFrame = self.dataFrame.loc[:, self.header_list[:-1]]
                self.dataFrame['mass'] = 4
            else:
                self.dataFrame = self.dataFrame.loc[:, self.header_list]

            self.dataFrame['pressure'] = round(self.dataFrame['pressure'].interpolate(), 1)
            self.dataFrame.drop_duplicates(subset='datetime', inplace=True)
            sample_rate = randrange(start=10, stop=self.max_data_rate)
            print(sample_rate)
            self.dataFrame = self.dataFrame.loc[::sample_rate, :]
            self.dataFrame = self.dataFrame[self.dataFrame.pressure.notnull()]

            self.dataFrame.to_csv(path_or_buf=self.file_path, mode='a', index=False, header=False)
            print('Balloon ' + serial_n + ' was dwonloaded')
            return self.file_path
        else:
            print("Couldnt download data.")
            return None
        
    def get_dataFrame(self):
        try:
            dataFrame = pd.read_csv(self.file_path)
            return dataFrame
        except:
            return None


if __name__ == '__main__':
    sonde = Sondehub_data_collector()
    sonde.download_data('U2060120' ,date.datetime.fromisoformat('2023-03-18'))