from cProfile import label
from curses.ascii import alt
from tkinter import Variable
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

        # lat layers
        self.a1 = nn.Tanh()
        self.l1_1 = nn.Linear(input_size, hidden_size)
        self.l1_2 = nn.Linear(hidden_size, 1)

        # lon layers
        self.a2 = nn.Tanh()
        self.l2_1 = nn.Linear(input_size, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, 1)

        # alt layers
        self.a3 = nn.Tanh()
        self.l3_1 = nn.Linear(input_size, hidden_size)
        self.l3_2 = nn.Linear(hidden_size, 1)

        

    def forward(self, x):

        out1 = self.l1_1(x)
        #out1 = self.a1(out1)
        out1 = self.l1_2(out1)

        out2 = self.l2_1(x)
        #out2 = self.a2(out2)
        out2 = self.l2_2(out2)

        out3 = self.l3_1(x)
        #out3 = self.a3(out3)
        out3 = self.l3_2(out3)

        return {'lat': out1, 'lon': out2, 'alt': out3}
    
    def train_step(model, data, optimizers, criterion):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(data):
            x_ = torch.autograd.Variable(inputs, requires_grad=True)
            y_ = torch.autograd.Variable(labels)


            def closure_lat():

                optimizers[0].zero_grad()

                y_pred = model(x_)

                lat_pred = y_pred['lat']

                loss1 = criterion(lat_pred, y_[0][0])

                loss1.backward()

                return loss1

            def closure_lon():

                optimizers[1].zero_grad()

                y_pred = model(x_)

                lon_pred = y_pred['lon']

                loss2 = criterion(lon_pred, y_[0][1])

                loss2.backward()

                return loss2
                

            def closure_alt():

                optimizers[2].zero_grad()

                y_pred = model(x_)

                alt_pred = y_pred['alt']

                loss3 = criterion(alt_pred, y_[0][2])

                loss3.backward()

                return loss3
            
            optimizers[0].step(closure_lat)
            optimizers[1].step(closure_lon)
            optimizers[2].step(closure_alt)

            loss = closure_lat() + closure_lon() + closure_alt()
            running_loss += loss.item()

        return running_loss

    

if __name__ == '__main__':

    hidden_size = 128
    learinging_rate = 0.003
    batch_size = 20
    num_epochs = 1000

    dataset_train = BalloonDataset(train=True)
    dataloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)

    dataset_test = BalloonDataset(train=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=0)

    dataiter = iter(dataloader)
    data = dataiter.next()
    features, target = data

    input_size =  features.shape[1]
    total_samples = len(dataset_train)


    pred_model = PredictionModel(features.shape[1], hidden_size)

    criterion = nn.L1Loss()
    optimizer1 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)
    optimizer2 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)
    optimizer3 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)

    for epoch in range(num_epochs):
        running_loss = PredictionModel.train_step(model=pred_model,
                              data=dataloader,
                              optimizers=[optimizer1, optimizer2, optimizer3],
                              criterion=criterion)
        print(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")

    
    with torch.no_grad():
        map_view = view.ViewMap()
        ecmwf = ecwfDC.ECMWF_data_collector()

        input_for_pred, _ = dataloader.dataset[0]
        lat_pred = dataset_test.get_lat(0)
        lon_pred = dataset_test.get_lon(0)
        alt_pred = dataset_test.get_alt(0)
        datetime_pred = datetime.datetime.fromisoformat('2023-03-09 10:51:44')
        lat_target = None
        lon_target = None
        alt_target = None
        predictions_n = 0

        for i in range(30):
            print('prediction: ', predictions_n)
            predictions_n += 1

            inputs, labels = dataloader.dataset[i]
            prediciton = pred_model(input_for_pred)
            
            lat = dataset_test.get_lat(i)
            lon = dataset_test.get_lon(i)
            alti = dataset_test.get_alt(i)

            lat_pred += prediciton['lat'].item() / 100
            lon_pred += prediciton['lon'].item() / 100
            alt_pred += prediciton['alt'].item() * 100
            pressure_pred = inputs[0].item() * 100
            mass_pred = inputs[1].item()
            datetime_pred += datetime.timedelta(seconds=180)

            temp_pred, wind_u_pred, wind_v_pred = ecmwf.get_data(lat_pred, lon_pred, pressure_pred, datetime_pred)

            temp_pred -= 273
            pressure_pred /= 100

            input_for_pred = torch.tensor([pressure_pred , mass_pred, temp_pred, wind_u_pred, wind_v_pred], dtype=torch.float32)

            lat_target = lat + labels[0].item() / 100
            lon_target = lon + labels[1].item() / 100
            alt_target = alti + labels[2].item() * 100

            map_view.add_point(lat_pred, lon_pred, False)
            map_view.add_point(lat_target, lon_target, True)

            print(f'Wind: {wind_u_pred}, {wind_v_pred}')
            print(f'Last target pos: {lat_target}, {lon_target}, {alt_target}')
            print(f'Last predict pos: {lat_pred}, {lon_pred}, {alt_pred}')

        
        map_view.show_map()

    
