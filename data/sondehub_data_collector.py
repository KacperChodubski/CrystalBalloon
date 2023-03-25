from random import random, randrange
from typing import Dict
import pandas as pd
import sondehub
import datetime as date
import os
import random




class Sondehub_data_collector:
    def __init__(self):
        self.header_list = ('serial','datetime','alt','lat', 'lon','vel_h', 'vel_v', 'pressure')
        self.cur_dir = os.path.dirname(__file__)
        self.dataFrame: pd.DataFrame = None
        self.max_data_rate = 90
        random.seed()
        self.collection = pd.DataFrame()

        
        self.file_path = os.path.join(self.cur_dir, "balloon/sondehub_datas.csv")

    def _convert_datatime(self, sonde_datetime: str):
        substr = sonde_datetime[:10] + '-' + sonde_datetime[11:19]
        datetime = date.datetime.fromisoformat(substr)
        return datetime
    
    # Start collecting data from recent flights on sondehub

    def collect_data(self, number_of_data):

        def on_message(message: Dict):
            if (message.get('pressure') is None):
                return
            message['datetime'] = self._convert_datatime(message['datetime'])
            data_of_baloon = pd.DataFrame([message], columns=self.header_list)
            data_of_baloon['mass'] = 4
            data_of_baloon['pressure'] = message['pressure']
            self.collection = pd.concat([self.collection, data_of_baloon], sort=False)
            print(f'downloaded: {len(self.collection)} / {number_of_data}')
            

        test = sondehub.Stream(on_message=on_message)
        test.loop_start()
        while len(self.collection) < number_of_data:
            pass
        test.loop_stop(force=False)
        self.collection['pressure'] = round(self.collection['pressure'].interpolate(), 1)
        self.collection.drop_duplicates(subset='datetime', inplace=True)
        sample_rate = randrange(start=10, stop=self.max_data_rate)
        self.collection = self.collection.loc[::sample_rate, :]
        self.collection = self.collection[self.collection.pressure.notnull()]
        print(self.collection)
        self.collection.to_csv(path_or_buf=self.file_path, mode='a', index=False, header=False)
        print(f'{number_of_data} frames was dwonloaded')


    # Download data from specific balloon or date from sondehub
    
    def download_data(self, serial_n: str = None, datetime: date.date = None):
        if (serial_n):
            frames = sondehub.download(serial=serial_n)
        else:
            frames = sondehub.download(datetime_prefix=datetime.isoformat())
        if (len(frames) > 0):
            self.dataFrame = pd.DataFrame.from_dict(frames, orient='columns')

            self.dataFrame['datetime'] = self.dataFrame['datetime'].apply(self._convert_datatime)

            self.dataFrame = self.dataFrame.loc[:, self.header_list]
            self.dataFrame['mass'] = 4

            self.dataFrame['pressure'] = round(self.dataFrame['pressure'].interpolate(), 1)
            self.dataFrame.drop_duplicates(subset='datetime', inplace=True)
            sample_rate = randrange(start=10, stop=self.max_data_rate)
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
    #sonde.download_data('U1830227' ,date.datetime.fromisoformat('2023-03-24'))
    sonde.collect_data(100)