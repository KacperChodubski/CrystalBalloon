from scipy.datasets import download_all
import torch
import torch.nn as nn
import utils.utils as utils
from model.model import PredictionModel
from torch.utils.tensorboard import SummaryWriter
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
                writer.add_scalar('Loss/Loss-lat', lat_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/Loss-lon', lon_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/Loss-alt', alt_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/Total-loss', total_loss.item(), len(train_loader) * epoch + batch_id + 1)

            if training_config['console_log_freq'] is not None and batch_id % training_config['console_log_freq'] == 0:
                print(f'time elapsed={(time.time()-ts)/60:.2f}[min]|epoch={epoch + 1}|batch=[{batch_id + 1}/{len(train_loader)}]|lat-loss={lat_loss / training_config["console_log_freq"]:.4f}|lon-loss={lon_loss / training_config["console_log_freq"]:.4f}|alt-loss={alt_loss / training_config["console_log_freq"]:.4f}|total loss={total_loss / training_config["console_log_freq"]:.4f}')
                lat_loss, lon_loss, alt_loss = [0., 0., 0.]
    
    training_state = {
        'state_dict': net.state_dict(),
        'optimazer_state': optimizer.state_dict(),
    }
    model_name = f'model_{training_config["model_name"]}.pth'
    torch.save(training_state, os.path.join(training_config['model_binaries_path'], model_name))

if __name__ == "__main__":
    dataset_up_path = os.path.join(os.path.dirname(__file__), 'data', 'balloon', 'datasets_up.csv')
    dataset_down_path = os.path.join(os.path.dirname(__file__), 'data', 'balloon', 'datasets_down.csv')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'trained_models', 'binaries')





    trainging_up_config = {
    'hidden_size': 8,
    'learinging_rate': 1e-3,
    'momentum': 0.99,
    'num_of_epochs': 100,
    'train_batch_size': 2,
    'test_batch_size': 32,
    'train_ratio': 0.8,
    'model_name': 'up',
    'dataset_path': dataset_up_path,
    'model_binaries_path': model_binaries_path,
    'enable_tensorboard': True,
    'console_log_freq': 400,
    }

    train(trainging_up_config)

    trainging_down_config = {
        'hidden_size': 8,
        'learinging_rate': 1e-3,
        'momentum': 0.99,
        'num_of_epochs': 100,
        'train_batch_size': 2,
        'test_batch_size': 32,
        'train_ratio': 0.8,
        'model_name': 'down',
        'dataset_path': dataset_down_path,
        'model_binaries_path': model_binaries_path,
        'enable_tensorboard': True,
        'console_log_freq': 400,
    }

    train(trainging_down_config)