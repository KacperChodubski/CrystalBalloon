import csv
from distutils.command.build_scripts import first_line_re
from email import header
from math import fabs
from queue import Empty
from tokenize import String
import numpy as np
import pandas as pd
import sondehub
import datetime as date
from dateutil import parser




class Sondehub_data_collector:
    def __init__(self):
        self.header_list = ('serial','datetime','alt','lat', 'lon','vel_h', 'vel_v', 'pressure', 'mass')
        self.directory_path = './'
        self.collect_train_data = True
        self.dataFrame: pd.DataFrame = None
        self.data_rate = 180

        
        self.test_file_path = './balloon/sondehub_datas_test.csv'
        self.train_file_path = './balloon/sondehub_datas.csv'

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
            self.dataFrame = self.dataFrame.loc[::self.data_rate, :]
            self.dataFrame = self.dataFrame[self.dataFrame.pressure.notnull()]

            file_path = None

            if self.collect_train_data:
                file_path = self.train_file_path
            else:
                file_path = self.test_file_path

            self.dataFrame.to_csv(file_path, mode='a', index=False, header=False)
            print('Balloon ' + serial_n + ' was dwonloaded')
            return file_path
        else:
            print("Couldnt download data.")
            return None
        
    def get_train_dataFrame(self):
        try:
            self.dataFrame = pd.read_csv(self.train_file_path)
            return self.dataFrame
        except:
            return None
        
    def get_test_dataFrame(self):
        try:
            self.dataFrame = pd.read_csv(self.test_file_path)
            return self.dataFrame
        except:
            return None
        


if __name__ == '__main__':
    sonde = Sondehub_data_collector()
    sonde.download_data('S5140565' ,date.datetime.fromisoformat('2023-03-10'))










