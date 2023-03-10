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
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.act6 = nn.Tanh()
        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.act7 = nn.Tanh()
        self.l8 = nn.Linear(hidden_size, hidden_size)
        self.act8 = nn.Tanh()
        self.l9 = nn.Linear(hidden_size, 3)
        

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
        out = self.act6(out)
        out = self.l7(out)
        out = self.act7(out)
        out = self.l8(out)
        out = self.act8(out)
        out = self.l9(out)
        return out
    

if __name__ == '__main__':

    hidden_size = 128
    learinging_rate = 0.03
    batch_size = 50
    num_epochs = 400

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
    optimizer = torch.optim.SGD(pred_model.parameters(), lr=learinging_rate)


    n_iterations = math.ceil(total_samples / batch_size)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = pred_model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 50 == 0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/ {n_iterations}, loss = {loss.item():.4f}')

    
    with torch.no_grad():
        map_view = view.ViewMap()
        dataiter = iter(dataloader_test)
        data = dataiter.next()
        features, target = data
        for i in range(0, len(dataloader_test), 300):
            inputs, labels = dataloader_test.dataset[i]
            if i == 30:
                break
            outputs = pred_model(inputs)

            
            lat = dataset_test.get_lat(i)
            lon = dataset_test.get_lon(i)
            alti = dataset_test.get_alt(i)

            
            #print(f'real out {lat + labels[0].item() / 100}, {lon + labels[1].item() / 100} , {alti + labels[2].item() * 1000}')
            #print(f'pred out {lat + outputs[0].item() / 100}, {lon + outputs[1].item() / 100}, {alti + outputs[2].item * 1000}')
            map_view.add_point(lat + labels[0].item() / 100,lon + labels[1].item() / 100, True)
            map_view.add_point(lat + outputs[0].item() / 100, lon + outputs[0].item() / 100, False)

        map_view.show_map()

    
