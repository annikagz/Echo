import torch
from torch import nn
import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statistics
from torch.utils.tensorboard import SummaryWriter
from networks.Network_Extras import init_weights
from itertools import groupby
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
from utility.Data_Processing import ReferenceTrajectoryData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TEMPORAL CONVOLUTIONAL NETWORK =======================================================================================
class TempConvNetwork(nn.Module):
    def __init__(self, n_inputs=1, kernel_size=5, stride=1, dilation=5, dropout=0.2, n_AE_layers=3):
        super(TempConvNetwork, self).__init__()
        # Here, define each layer with their inputs, for example:
        self.input_size = n_inputs
        self.flattened_length = 1920
        self.AE_layers = n_AE_layers

        self.TCN = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=8, kernel_size=kernel_size, stride=stride,
                      dilation=dilation, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(int(self.flattened_length), int(self.flattened_length / 2)),  # (1, 512)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 2), int(self.flattened_length / 4)),  # (1, 256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 4), int(self.flattened_length / 8)),  # (1, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 8), int(self.flattened_length / 16)),  # (1, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.flattened_length / 16), 1)
            )

    def forward(self, EMG_signal):
        out = self.TCN(EMG_signal)
        return out


class RunTCN:
    def __init__(self, n_channels, epochs, saved_model_name, load_model=False,
                 initial_lr=0.001):
        self.model = TempConvNetwork(n_inputs=n_channels, kernel_size=5, stride=1, dilation=4, dropout=0.4).to(device)
        self.model_type = 'TCN'
        self.saved_model_name = saved_model_name
        self.saved_model_path = '/media/ag6016/Storage/Echo/SavedModels/' + self.saved_model_name + '.pth'
        self.model.TCN.apply(init_weights)
        if load_model:
            self.model.load_state_dict(torch.load(self.saved_model_path))
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, betas=(0.9, 0.999),
                                          weight_decay=initial_lr * 0.1)
        self.epochs = epochs
        self.writer = SummaryWriter()
        self.recorded_training_error = 100
        self.recorded_validation_error = 100
        self.recorded_testing_error = None
        self.epochs_ran = 0
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def train_network(self, x_train, y_train, x_val, y_val):
        self.x_train, self.y_train, self.x_val, self.y_val = x_train, y_train, x_val, y_val

        rep_step = 0
        lowest_error = 1000.0
        cut_off_counter = 0
        for epoch in range(self.epochs):
            print("Epoch number:", epoch)
            running_training_loss = 0.0
            running_validation_loss = 0.0
            for rep in tqdm(np.arange(self.x_train.shape[0])):
                x_train = self.x_train[rep, :, :, :].to(device)
                y_train = self.y_train[rep, :, :, :].to(device)
                predicted = self.model.forward(EMG_signal=x_train.float())
                loss = self.criterion(torch.squeeze(predicted), torch.squeeze(y_train.float()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_training_loss += loss.item()
                rep_step += 1
            recorded_training_error = math.sqrt(running_training_loss / (self.x_train.shape[-1]))
            self.writer.add_scalar("Epoch training loss ", recorded_training_error, global_step=epoch)
            # VALIDATION LOOP
            with torch.no_grad():
                for rep in range(self.x_val.shape[0]):
                    x_val = self.x_val[rep, :, :, :].to(device)
                    y_val = self.y_val[rep, :, :, :].to(device)
                    predicted = self.model.forward(EMG_signal=x_val.float())
                    validation_loss = self.criterion(torch.squeeze(predicted), torch.squeeze(y_val.float()))
                    running_validation_loss += validation_loss.item()
            recorded_validation_error = math.sqrt(running_validation_loss / (self.x_val.shape[-1]))
            self.writer.add_scalar("Epoch val loss ", recorded_validation_error, global_step=epoch)
            if recorded_validation_error < lowest_error:
                torch.save(self.model.state_dict(), self.saved_model_path)
                lowest_error = recorded_validation_error
                self.recorded_validation_error = recorded_validation_error
                self.recorded_training_error = recorded_training_error
                print("The errors are ", self.recorded_training_error, self.recorded_validation_error)
                self.epochs_ran = epoch
                cut_off_counter = 0
                print("it's lower")
            else:
                cut_off_counter += 1
            if cut_off_counter > 10:
                break

    def test_network(self, x_test_data, y_test_data):
        running_loss = 0.0
        with torch.no_grad():
            for rep in range(x_test_data.shape[0]):
                x_test = x_test_data[rep, :, :, :].to(device)
                y_test = y_test_data[rep, :, :, :].to(device)
                predicted = self.model.forward(EMG_signal=x_test.float())
                loss = self.criterion(torch.squeeze(predicted), torch.squeeze(y_test.float()))
                running_loss += loss.item()
        recorded_error = math.sqrt(running_loss / (x_test_data.shape[-1]))
        self.recorded_testing_error = recorded_error

