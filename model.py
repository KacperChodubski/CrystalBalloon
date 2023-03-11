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
        self.l1_2 = nn.Linear(hidden_size, hidden_size)
        self.l1_3 = nn.Linear(hidden_size, 1)

        # lon layers
        self.a2 = nn.Tanh()
        self.l2_1 = nn.Linear(input_size, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, hidden_size)
        self.l2_3 = nn.Linear(hidden_size, 1)

        # alt layers
        self.a3 = nn.Tanh()
        self.l3_1 = nn.Linear(input_size, hidden_size)
        self.l3_2 = nn.Linear(hidden_size, hidden_size)
        self.l3_3 = nn.Linear(hidden_size, 1)

        

    def forward(self, x):

        out1 = self.l1_1(x)
        out1 = self.a1(out1)
        out1 = self.l1_2(out1)
        out1 = self.a1(out1)
        out1 = self.l1_3(out1)

        out2 = self.l2_1(x)
        out2 = self.a2(out2)
        out2 = self.l2_2(out2)
        out2 = self.a2(out2)
        out2 = self.l2_3(out2)

        out3 = self.l3_1(x)
        out3 = self.a3(out3)
        out3 = self.l3_2(out3)
        out3 = self.a3(out3)

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

                optimizers[0].zero_grad()

                y_pred = model(x_)

                lon_pred = y_pred['lon']

                loss2 = criterion(lon_pred, y_[0][1])

                loss2.backward()

                return loss2
                

            def closure_alt():

                optimizers[0].zero_grad()

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

    hidden_size = 32
    learinging_rate = 0.03
    batch_size = 1
    num_epochs = 300

    dataset = BalloonDataset()
    
    train_size = round(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    dataiter = iter(train_loader)
    data = dataiter.next()
    features, target = data

    input_size =  features.shape[1]
    num_training_steps = len(train_loader) * num_epochs

    pred_model = PredictionModel(features.shape[1], hidden_size)

    criterion = nn.MSELoss()
    optimizer1 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)
    optimizer2 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)
    optimizer3 = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)

    for epoch in range(num_epochs):
        running_loss = PredictionModel.train_step(model=pred_model,
                              data=train_loader,
                              optimizers=[optimizer1, optimizer2, optimizer3],
                              criterion=criterion)
        print(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")

    
    # with torch.no_grad():
    #     map_view = view.ViewMap()
    #     ecmwf = ecwfDC.ECMWF_data_collector()

    #     input_for_pred, _ = test_loader.dataset[0]
        
    #     lat_target = None
    #     lon_target = None
    #     alt_target = None
    #     predictions_n = 0

    #     lat_pred = test_set[0]
    #     lon_pred = test_set[0]
    #     alt_pred = test_set[0]

    #     for i in range(35):
    #         print('prediction: ', predictions_n)
    #         predictions_n += 1

    #         inputs, labels = test_loader.dataset[i]
    #         prediciton = pred_model(input_for_pred)
            
    #         lat = test_set.get_lat(i)
    #         lon = test_set.get_lon(i)
    #         alti = test_set.get_alt(i)

    #         lat_pred += prediciton['lat'].item()
    #         lon_pred += prediciton['lon'].item()
    #         alt_pred += prediciton['alt'].item()

            
    #         input_for_pred = torch.tensor(inputs, dtype=torch.float32)

    #         lat_target = lat + labels[0].item()
    #         lon_target = lon + labels[1].item()
    #         alt_target = alti + labels[2].item()

    #         map_view.add_point(lat_pred, lon_pred, False)
    #         map_view.add_point(lat_target, lon_target, True)

        
    #     map_view.show_map()

    
