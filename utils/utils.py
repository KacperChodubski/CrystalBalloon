import math
import datetime
import torch
import data.ecmwf_data_collector
from torch.utils.data import DataLoader, Dataset
import numpy as np

DATA_MEAN = 86.6239
DATA_STD = 151.8148

class BalloonDataset(Dataset):
    def __init__(self,path, normalize = True):
        self.path = path
        self.normalize = normalize

        xy = np.loadtxt(self.path, delimiter=',', dtype=np.float32)

        self.x = xy[:, :-3]
        
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(xy[:, -3:])

        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        inputs = self.x[index]
        if self.normalize:
            inputs = (self.x[index] - DATA_MEAN) / DATA_STD
        return inputs, self.y[index]

    def __len__(self):
        return self.n_samples


def find_bd(balloon_mass: str):
        burst_diameters = {
            'h100': 180,
            'h300': 380,
            'h350': 410,
            'h500': 500,
            'h600': 580,
            'h800': 680,
            'h1000': 750,
            'h1200': 850,
            'h1600': 1050,
            'h2000': 1100,
            'h3000': 12.50,
        }
        bd = burst_diameters[balloon_mass] / 100
        return bd
    
def find_cd(balloon_mass: str):
    cds = {
        "h200": 0.25,
        "h100": 0.25,
        "h300": 0.25,
        "h350": 0.25,
        "h500": 0.25,
        "h600": 0.30,
        "h750": 0.30,
        "h800": 0.30,
        "h950": 0.30,
        "h1000": 0.30,
        "h1200": 0.25,
        "h1500": 0.25,
        "h1600": 0.25,
        "h2000": 0.25,
        "h3000": 0.25,
    }
    cd = cds[balloon_mass]
    return cd

def calculate_burst_altitude(balloon_mass: str, payload_mass: float, ascent_rate: float):

        rho_g = 0.1786 # for hel

        
        mb = float(balloon_mass[1:]) / 1000
        mp = payload_mass / 1000
        

        bd = find_bd(balloon_mass) 
        cd = find_cd(balloon_mass) 
        rho_a = 1.2050                  # air density
        adm = 7238.3                    # air density model
        ga = 9.80665                    # gravity aceleration
        tar = ascent_rate               # target ascenting rate


        burst_volume = 4/3 * math.pi * (bd/2)**3

        a = ga * (rho_a - rho_g) * (4.0 / 3.0) * math.pi
        b = -0.5 * math.pow(tar, 2) * cd * rho_a * math.pi
        c = 0
        d = - (mp + mb) * ga
        f = (((3*c)/a) - (math.pow(b, 2) / math.pow(a,2)) / 3.0)
        g = (((2*math.pow(b,3))/math.pow(a,3)) - ((9*b*c)/(math.pow(a,2))) + ((27*d)/a)) / 27.0
        h = (math.pow(g,2) / 4.0) + (math.pow(f,3) / 27.0)
        R = (-0.5 * g) + math.sqrt(h)
        S = math.pow(R, 1.0/3.0)
        T = (-0.5 * g) - math.sqrt(h)
        U = math.pow(T, 1.0/3.0)

        launch_radius = (S+U) - (b/(3*a))

        launch_volume = (4.0/3.0) * math.pi * math.pow(launch_radius, 3)

        volume_ratio = launch_volume / burst_volume

        burst_altitude = -(adm) * math.log(volume_ratio)

        print(f'Predicted burst altitude: {burst_altitude}')
        print(f'Required launch volume: {launch_volume}')

        return burst_altitude

def calculate_pressure(ecmwf: data.ecmwf_data_collector.ECMWF_data_collector, lat, lon, alt, alt0, time_for_pred: datetime.date):
    Tb, _, _ = ecmwf.get_data(lat, lon, 1000, time_for_pred)
    msl = ecmwf.get_msl(lat, lon, time_for_pred)
    Lb = -0.0065
    g0 = 9.80665
    R = 8.31432
    M = 0.0289644
    pressure: float
    if alt < 10000:
        pressure = msl * (1 + (Lb/Tb)*(alt - alt0))**((-g0 * M)/(R*Lb))
    elif alt > 10000:
        pressure = msl * math.exp((-g0 * M * (alt - alt0)) / (R*Tb))

    return pressure

def prepare_data(input, device, batch_size = 1, should_normalize = True):
     input = torch.Tensor(input)
     input = (input - DATA_MEAN) / DATA_STD
     input = input.to(device)
     input.repeat(batch_size, 1, 1, 1)
     return input


def get_data_loaders(trainging_config, should_normalize = True):
    dataset = BalloonDataset(trainging_config['dataset_path'], should_normalize)

    train_size = round(len(dataset) * trainging_config['train_ratio'])
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=trainging_config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=trainging_config['test_batch_size'], shuffle=False)

    return train_loader, test_loader


def calculate_mean_std(train_loader, train_set, feature_size):
    num_of_features = len(train_set) * feature_size

    total_sum = 0
    for batch in train_loader: total_sum += batch[0].sum()
    mean = total_sum / num_of_features

    sum_of_squared_error = 0
    for batch in train_loader: sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
    std = torch.sqrt(sum_of_squared_error / num_of_features)

    return mean, std
     