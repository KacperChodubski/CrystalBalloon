import torch
import torch.nn as nn
from data.data_module import BalloonDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import view
import data.ecmwf_data_collector as ecwfDC
import datetime
#from torch.utils.tensorboard import SummaryWriter
import sys

class PredictionModel(nn.Module):
    def __init__(self, dataset):

        # hyperparameters
        hidden_size = 32
        learinging_rate = 1e-3
        batch_size = 16

        # Loading data
    
        train_size = round(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        train_set, self.test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)

        # Setting up input size

        dataiter = iter(self.train_loader)
        data = dataiter.next()
        features, _ = data

        input_size =  features.shape[1]

        super(PredictionModel, self).__init__()

        #self.writer = SummaryWriter("runs/logs")

        # lat layers
        self.l1_1 = nn.Linear(input_size, hidden_size)
        self.l1_2 = nn.Linear(hidden_size, hidden_size)
        self.l1_3 = nn.Linear(hidden_size, hidden_size)
        self.l1_4 = nn.Linear(hidden_size, 1)

        # lon layers
        self.l2_1 = nn.Linear(input_size, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, hidden_size)
        self.l2_3 = nn.Linear(hidden_size, hidden_size)
        self.l2_4 = nn.Linear(hidden_size, 1)

        # alt layers
        self.l3_1 = nn.Linear(input_size, hidden_size)
        self.l3_2 = nn.Linear(hidden_size, hidden_size)
        self.l3_3 = nn.Linear(hidden_size, hidden_size)
        self.l3_4 = nn.Linear(hidden_size, 1)

        # Setting up optimizer and criterion

        self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=learinging_rate, momentum=0.99)
        self.optimizer = torch.optim.Adam(self.parameters(), lr= learinging_rate)

        

    def forward(self, x):

        out1 = self.l1_1(x)
        out1 = self.l1_2(out1)
        out1 = self.l1_3(out1)
        out1 = self.l1_4(out1)

        out2 = self.l2_1(x)
        out2 = self.l2_2(out2)
        out2 = self.l2_3(out2)
        out2 = self.l2_4(out2)

        out3 = self.l3_1(x)
        out3 = self.l3_2(out3)
        out3 = self.l3_3(out3)
        out3 = self.l3_4(out3)

        return {'lat': out1, 'lon': out2, 'alt': out3}
    
    def train_step(self):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(self.train_loader):

            def closure_lat():
                self.optimizer.zero_grad()

                y_pred = self(inputs)
                lat_pred = y_pred['lat']

                target = torch.FloatTensor(labels[: , 0])
                target = torch.unsqueeze(target, 1)

                loss1 = self.criterion(lat_pred, target)
                loss1.backward()

                return loss1

            def closure_lon():
                self.optimizer.zero_grad()

                y_pred = self(inputs)
                lon_pred = y_pred['lon']

                target = torch.FloatTensor(labels[: , 1])
                target = torch.unsqueeze(target, 1)

                loss2 = self.criterion(lon_pred, target)
                loss2.backward()

                return loss2

            def closure_alt():
                self.optimizer.zero_grad()

                y_pred = self(inputs)
                alt_pred = y_pred['alt']

                target = torch.FloatTensor(labels[: , 2])
                target = torch.unsqueeze(target, 1)

                loss3 = self.criterion(alt_pred, target)
                loss3.backward()
                return loss3

                
            
            self.optimizer.step(closure_lat)
            self.optimizer.step(closure_lon)
            self.optimizer.step(closure_alt)

            loss = closure_lat() + closure_lon() + closure_alt()
            running_loss += loss.item()

        return running_loss
    
    def train(self, num_epochs):
        total_steps = len(self.train_loader) * num_epochs
        for epoch in range(num_epochs):
            running_loss = self.train_step()
            print(f"Epoch: {epoch + 1:02}/{num_epochs} Loss: {running_loss:.5e}")

    def validation(self):
        with torch.no_grad():
            map_view = view.ViewMap()
            ecmwf = ecwfDC.ECMWF_data_collector()

            input_for_pred, _ = self.test_loader.dataset[0]
            
            predictions_n = 0

            lat_start = 54.70527
            lon_start = 17.65194
            alt_start = 2974.15

            lat_pred = lat_start
            lon_pred = lon_start
            alt_pred = alt_start

            lat_target = lat_start
            lon_target = lon_start
            alt_target = alt_start

            for i in range(60):
                print('prediction: ', predictions_n)
                predictions_n += 1

                inputs, labels = self.test_loader.dataset[i]
                prediciton = pred_model(input_for_pred)
                
                input_for_pred = torch.tensor(inputs, dtype=torch.float32)        

                map_view.add_point(lat_pred, lon_pred, False)
                map_view.add_point(lat_target, lon_target, True)

                lat_pred += prediciton['lat'].item() / 100
                lon_pred += prediciton['lon'].item() / 100
                alt_pred += prediciton['alt'].item() * 1000

                lat_target += labels[0].item() / 100
                lon_target += labels[1].item() / 100 
                alt_target += labels[2].item() * 1000

                print(f'Target pos {lat_target}, {lon_target}, {alt_target}')
                print(f'Predic pos {lat_pred}, {lon_pred}, {alt_pred}')

            
            map_view.show_map()


    

if __name__ == '__main__':

    dataset = BalloonDataset()

    pred_model = PredictionModel(dataset=dataset)
    num_epochs = 50
    pred_model.train(num_epochs)
    pred_model.validation()