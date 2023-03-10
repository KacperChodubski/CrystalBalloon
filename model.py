from cProfile import label
from curses.ascii import alt
from tkinter.tix import Tree
from turtle import forward
from numpy import dtype
import torch
import torch.nn as nn
from data.data_module import BalloonDataset
from torch.utils.data import DataLoader
import math
import view
import data.ecmwf_data_collector as ecwfDC
import datetime

class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PredictionModel, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.Tanh()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.Tanh()
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.act4 = nn.Tanh()
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.act5 = nn.Tanh()
        self.l6 = nn.Linear(hidden_size, 3)
        

    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        out = self.act2(out)
        out = self.l3(out)
        out = self.act3(out)
        out = self.l4(out)
        out = self.act4(out)
        out = self.l5(out)
        out = self.act5(out)
        out = self.l6(out)
        return out
    

if __name__ == '__main__':

    hidden_size = 64
    learinging_rate = 0.001
    batch_size = 10
    num_epochs = 800

    dataset_train = BalloonDataset(train=True)
    dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_test = BalloonDataset(train=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    data = dataiter.next()
    features, target = data

    input_size =  features.shape[1]
    total_samples = len(dataset_train)


    pred_model = PredictionModel(features.shape[1], hidden_size)

    criterion = nn.SmoothL1Loss() # jeszcze do testowania
    optimizer = torch.optim.ASGD(pred_model.parameters(), lr=learinging_rate)


    n_iterations = math.ceil(total_samples / batch_size)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = pred_model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/ {n_iterations}, loss = {loss.item():.4f}')

    
    with torch.no_grad():
        map_view = view.ViewMap()
        ecmwf = ecwfDC.ECMWF_data_collector()

        input_for_pred, _ = dataloader_test.dataset[0]
        lat_pred = dataset_test.get_lat(0)
        lon_pred = dataset_test.get_lon(0)
        alt_pred = dataset_test.get_alt(0)
        datetime_pred = datetime.datetime.fromisoformat('2023-03-09 23:21:44')
        lat_target = None
        lon_target = None
        alt_target = None
        predictions_n = 0

        for i in range(len(dataloader_test)):
            print('prediction: ', predictions_n)
            predictions_n += 1

            inputs, labels = dataloader_test.dataset[i]
            prediciton = pred_model(input_for_pred)
            
            lat = dataset_test.get_lat(i)
            lon = dataset_test.get_lon(i)
            alti = dataset_test.get_alt(i)

            lat_pred += prediciton[0].item()
            lon_pred += prediciton[1].item()
            alt_pred += prediciton[2].item()
            pressure_pred = inputs[0].item()
            mass_pred = inputs[1].item()
            datetime_pred += datetime.timedelta(seconds=180)

            temp_pred, wind_u_pred, wind_v_pred = ecmwf.get_data(lat_pred, lon_pred, pressure_pred, datetime_pred)

            input_for_pred = torch.tensor([pressure_pred, mass_pred, temp_pred, wind_u_pred, wind_v_pred], dtype=torch.float32)
            
            #print(f'real out {lat + labels[0].item() / 100}, {lon + labels[1].item() / 100} , {alti + labels[2].item() * 1000}')
            #print(f'pred out {lat + outputs[0].item() / 100}, {lon + outputs[1].item() / 100}, {alti + outputs[2].item * 1000}')

            lat_target = lat + labels[0].item()
            lon_target = lon + labels[1].item()
            alt_target = alti + labels[2].item()

            map_view.add_point(lat_target, lon_target, True)
            map_view.add_point(lat_pred, lon_pred, False)

        print(f'Last target pos: {lat_target}, {lon_target}, {alt_target}')
        print(f'Last predict pos: {lat_pred}, {lon_pred}, {alt_pred}')
        map_view.show_map()

    
