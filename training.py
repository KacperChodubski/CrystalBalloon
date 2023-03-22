import torch
import torch.nn as nn
from zmq import device
import utils.utils as utils
import utils.view as view
from model.model import PredictionModel
import data.ecmwf_data_collector as ecwfDC
import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import math
import time
import os




def train(training_config):
    writer = SummaryWriter() # (tensorboard) writer will write to ./runs/ directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data loader
    train_loader, _ = utils.get_data_loaders(trainging_config=training_config)

    # prepare neural networks
    net = PredictionModel().train().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr= training_config['learinging_rate'])
    criterion = nn.MSELoss()

    acc_lat_loss, acc_lon_loss, acc_alt_loss = [0., 0., 0.]
    ts = time.time()

    for epoch in range(training_config['num_of_epochs']):
         #for samples in self.train_loader_norm:
         #     inputs, labels = samples
         for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            y_hat = net(inputs)



            # latitude loss
                
            lat_pred = y_hat['lat']

            target = torch.FloatTensor(labels[: , 0])
            target = torch.unsqueeze(target, 1)

            lat_loss = criterion(lat_pred, target)

            # longtitude loss
            
            lon_pred = y_hat['lon']

            target = torch.FloatTensor(labels[: , 1])
            target = torch.unsqueeze(target, 1)

            lon_loss = criterion(lon_pred, target)

            # altitude loss

            alt_pred = y_hat['alt']

            target = torch.FloatTensor(labels[: , 2])
            target = torch.unsqueeze(target, 1)

            alt_loss = criterion(alt_pred, target)
            
            total_loss = lat_loss + lon_loss + alt_loss
            total_loss.backward()

            optimizer.step()
            
            optimizer.zero_grad()

            acc_lat_loss += lat_loss.item()
            acc_lon_loss += lon_loss.item()
            acc_alt_loss += alt_loss.item()

            if training_config['enable_tensorboard']:
                writer.add_scalar('Loss_lat', lat_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss_lon', lon_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss_alt', alt_loss.item(), len(train_loader) * epoch + batch_id + 1)

            if training_config['console_log_freq'] is not None and batch_id % training_config['console_log_freq'] == 0:
                print(f'time elapsed={(time.time()-ts)/60:.2f}[min]|epoch={epoch + 1}|batch=[{batch_id + 1}/{len(train_loader)}]|lat-loss={lat_loss / training_config["console_log_freq"]}|lon-loss={lon_loss / training_config["console_log_freq"]}|alt-loss={alt_loss / training_config["console_log_freq"]}|total loss={(lat_loss + lon_loss + alt_loss) / training_config["console_log_freq"]}')
                lat_loss, lon_loss, alt_loss = [0., 0., 0.]
    
    training_state = {
        'state_dict': net.state_dict(),
        'optimazer_state': optimizer.state_dict(),
    }
    model_name = f'model_{str(time.gmtime())}.pth'
    torch.save(training_state, os.path.join(training_config['model_binaries_path'], model_name))


# def validation():
#         with torch.no_grad():
#             map_view = view.ViewMap()
#             ecmwf = ecwfDC.ECMWF_data_collector()

#             input_for_pred, _ = test_loader.dataset[0]
            
#             predictions_n = 0

#             lat_start = 54.70527
#             lon_start = 17.65194
#             alt_start = 2974.15

#             lat_pred = lat_start
#             lon_pred = lon_start
#             alt_pred = alt_start

#             lat_target = lat_start
#             lon_target = lon_start
#             alt_target = alt_start

#             for i in range(60):
#                 print('prediction: ', predictions_n)
#                 predictions_n += 1

#                 inputs, labels = self.test_loader.dataset[i]
#                 prediciton = pred_model(input_for_pred)
                
#                 input_for_pred = torch.tensor(inputs, dtype=torch.float32)

#                 map_view.add_point(lat_pred, lon_pred, False)
#                 map_view.add_point(lat_target, lon_target, True)

#                 lat_pred += prediciton['lat'].item() / 100
#                 lon_pred += prediciton['lon'].item() / 100
#                 alt_pred += prediciton['alt'].item() * 1000

#                 lat_target += labels[0].item() / 100
#                 lon_target += labels[1].item() / 100 
#                 alt_target += labels[2].item() * 1000

#                 print(f'Target pos {lat_target}, {lon_target}, {alt_target}')
#                 print(f'Predic pos {lat_pred}, {lon_pred}, {alt_pred}')

            
#             map_view.show_map()

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'balloon', 'datasets_up.csv')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'trained_models', 'binaries')





    trainging_up_config = {
    'hidden_size': 8,
    'learinging_rate': 1e-3,
    'momentum': 0.9,
    'num_of_epochs': 20,
    'train_batch_size': 2,
    'test_batch_size': 32,
    'train_ratio': 0.8,
    'dataset_path': dataset_path,
    'model_binaries_path': model_binaries_path,
    'enable_tensorboard': True,
    'console_log_freq': 100,
    }

    train(trainging_up_config)

    trainging_down_config = {
        'hidden_size': 8,
        'learinging_rate': 1e-3,
        'momentum': 0.9,
        'num_of_epochs': 20,
        'train_batch_size': 2,
        'test_batch_size': 32,
        'dataset_path': "data/balloon/datasets_down.csv",
        'enable_tensorboard': True,
        'log_ferq': 20,
    }