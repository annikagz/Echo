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


class Regressor(nn.Module):
    def __init__(self, in_channels=1, window_length=250, output_size=1, kernel_size=5, stride=1, dilation=2):
        super(Regressor, self).__init__()
        self.window_length = window_length
        self.in_channels = in_channels
        self.hidden_channels = 4

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels*2, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=self.hidden_channels*2, out_channels=self.hidden_channels*4, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(976, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_size),
            nn.Sigmoid()
        )

    def forward(self, input_window_data):
        out = self.model(input_window_data) * 100
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels=1, window_length=250, output_size=6, kernel_size=5, stride=1, dilation=2):
        super(Classifier, self).__init__()
        self.window_length = window_length
        self.in_channels = in_channels
        self.hidden_channels = 4
        if self.window_length == 250:
            self.flattened = 480
        elif self.window_length == 500:
            self.flattened = 976
        elif self.window_length == 1000:
            self.flattened = 1984

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels*2, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=self.hidden_channels*2, out_channels=self.hidden_channels*4, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, padding='same'),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(self.flattened, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_window_data):
        out = self.model(input_window_data)
        return out


class TrajectoryMappingNetwork:
    def __init__(self, learning_rate, epochs, saved_model_name, model_type='classifier', window_length=250,
                 number_of_classes=10, load=False):
        """
        This class focuses on training the model to map the healthy leg trajectory onto the gait cycle
        :param learning_rate: value of the learning rate used to train the model, given as a float
        :param epochs: number of epochs used to train the model, given as an int
        :param saved_model_name: string that will be used to save the model
        :param model_type: either a 'classifier' or a 'regressor', depending on the labels that we are using
        """
        self.LR = learning_rate
        self.epochs = epochs
        self.saved_name = saved_model_name
        self.saved_model_path = '/media/ag6016/Storage/Echo/SavedModels/' + saved_model_name + '.pth'
        self.model_type = model_type
        if self.model_type == 'classifier':
            self.model = Classifier(window_length=window_length, output_size=number_of_classes).to(device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, betas=(0.9, 0.999),
                                              weight_decay=self.LR*0.1)
            self.criterion = nn.CrossEntropyLoss().to(device)
        elif self.model_type == 'regressor':
            self.model = Regressor(window_length=window_length).to(device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR, betas=(0.9, 0.999),
                                              weight_decay=self.LR*0.1)
            self.criterion = nn.MSELoss().to(device)
        if load:
            self.model.load_state_dict(torch.load(self.saved_model_path))
        self.writer = None
        self.recorded_validation_error = None
        self.recorded_training_error = None
        self.recorded_test_error = None
        self.epochs_ran = None
        self.test_accuracy = None

        # For the update
        self.recorded_class_start_value = None
        self.current_class = None

    def train_network(self, x_train, y_train, x_val, y_val):
        # INITIALISE THE WEIGHTS
        self.writer = SummaryWriter()
        self.model.model.apply(init_weights)
        lowest_validation_error = 100000.0
        counter = 0
        for epoch in range(self.epochs):
            print("Epoch number ", epoch)
            running_train_loss = 0.0
            running_val_loss = 0.0
            for rep in tqdm(list(range(x_train.shape[0]))):
                training_y = y_train[rep, :, :, :].to(device)
                if self.model_type == 'classifier':
                    training_x = x_train[rep, :, :, :].to(device)
                    predicted = self.model.forward(training_x.float())
                    loss = self.criterion(predicted, torch.squeeze(training_y.float()))
                elif self.model_type == 'regressor':
                    training_x = x_train[rep, :, :, :].to(device)
                    predicted = self.model.forward(training_x.float())
                    loss = self.criterion(predicted.squeeze(), training_y.float().squeeze())
                else:
                    raise ValueError('Wrong model type')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
            recorded_training_error = math.sqrt(running_train_loss / x_train.shape[0])
            self.writer.add_scalar("Epoch training loss", recorded_training_error, global_step=epoch)
            # VALIDATION LOOP
            with torch.no_grad():
                for rep in range(x_val.shape[0]):
                    validation_y = y_val[rep, :, :, :].to(device)
                    if self.model_type == 'classifier':
                        validation_x = x_val[rep, :, :, :].to(device)
                        predicted = self.model.forward(validation_x.float())
                        loss = self.criterion(predicted, torch.squeeze(validation_y.float()))
                    elif self.model_type == 'regressor':
                        validation_x = x_val[rep, :, :, :].to(device)
                        predicted = self.model.forward(validation_x.float())
                        loss = self.criterion(predicted.squeeze(), validation_y.float().squeeze())
                    else:
                        raise ValueError('Wrong model type')
                    running_val_loss += loss.item()
            recorded_validation_error = math.sqrt(running_val_loss/x_val.shape[0])
            self.writer.add_scalar("Epoch validation loss", recorded_validation_error, global_step=epoch)
            if recorded_validation_error < lowest_validation_error:
                torch.save(self.model.state_dict(), self.saved_model_path)
                lowest_validation_error = recorded_validation_error
                self.recorded_validation_error = recorded_validation_error
                self.recorded_training_error = recorded_training_error
                self.epochs_ran = epoch
                print("The computed error is lower")
                counter = 0
            else:
                counter += 1
            if counter > 20:
                break

    def test_network(self, x_test, y_test, plot_output=True):
        running_test_loss = 0.0
        with torch.no_grad():
            prediction = []
            true = []
            for rep in range(x_test.shape[0]):
                testing_y = y_test[rep, :, :, :].to(device)
                if self.model_type == 'classifier':
                    testing_x = x_test[rep, :, :, :].to(device)
                    predicted = self.model(testing_x.float())
                    prediction.append(torch.argmax(predicted).cpu().detach().item())
                    true.append(torch.argmax(testing_y.float()).cpu().detach().item())
                    loss = self.criterion(predicted, torch.squeeze(testing_y.float(), dim=1))
                elif self.model_type == 'regressor':
                    testing_x = x_test[rep, :, :, :].to(device)
                    predicted = self.model.forward(testing_x.float())
                    loss = self.criterion(predicted.squeeze(), testing_y.float().squeeze())
                    prediction.append(predicted.cpu().detach().item())
                    true.append(testing_y.float().cpu().detach().item())
                else:
                    raise ValueError('Wrong model type')
                running_test_loss += loss.item()
            self.recorded_test_error = math.sqrt(running_test_loss / x_test.shape[0])
        if self.model_type == 'classifier':
            self.test_accuracy = sum(1 for x,y in zip(prediction,true) if x == y) / len(prediction)
            print("The testing accuracy of the model is ", self.test_accuracy)
        elif self.model_type == 'regressor':
            print("The final RMSE of the model is ", self.recorded_test_error)
        if plot_output:
            # plt.subplot(2, 1, 1)
            plt.plot(prediction[0:10000], label='Predicted')
            plt.plot(true[0:10000], label='True')
            plt.legend()
            # x_test = np.reshape(x_test.cpu().detach().numpy(), (-1, 1), order='F')
            # plt.subplot(2, 1, 2)
            # plt.plot(x_test[0:10000])
            plt.savefig(
                '/media/ag6016/Storage/Echo/Images/model_performance_' + self.saved_name + '.pdf', dpi=200,
                bbox_inches='tight')


class ReferenceTrajectoryUpdate:
    def __init__(self, initial_reference_trajectory, initial_reference_label, trained_model_name, window_step,
                 model, number_of_classes=10):
        self.initial_reference_trajectory = initial_reference_trajectory
        self.initial_reference_label = initial_reference_label
        self.trained_model_name = trained_model_name
        self.window_step = window_step
        self.number_of_classes = number_of_classes
        self.segmented_reference_trajectory = np.array([[c for _, c in g] for _, g
                                                        in groupby(zip(self.initial_reference_label,
                                                                       self.initial_reference_trajectory),
                                                                   key=lambda x: x[0])])
        self.number_of_classes = self.segmented_reference_trajectory.shape[0]
        self.recorded_first_values = list(np.zeros((self.number_of_classes, 1)))#self.segmented_reference_trajectory[:, 0, 0]
        self.recorded_slopes = None  # Need to add this so that the slope of the curve is the same
        self.model = model
        self.load_model()
        self.current_class = None

    def plot_gait_cycle_segmentation(self):
        for i in range(self.segmented_reference_trajectory.shape[0]):
            x = np.arange(i*self.segmented_reference_trajectory.shape[1],
                          i*self.segmented_reference_trajectory.shape[1]+self.segmented_reference_trajectory.shape[1])
            plt.plot(x, self.segmented_reference_trajectory[i, :, 0].squeeze(), linewidth=3)

    def load_model(self):
        trained_model_path = '/media/ag6016/Storage/Echo/SavedModels/' + self.trained_model_name + '.pth'
        self.model.load_state_dict(torch.load(trained_model_path))
        self.model.eval()

    def update_reference_trajectory_classification(self, valid_knee_angle, window_step, plot_continuously=False):
        """
        This is the function that will update the reference trajectory. This function assumes classification so that it
        is easier to write and run through.
        We want to add an error margin which will be able to get rid of outliers
        We also want to make sure that the first class is not counted, in case it doesn't start at the start of the gait phase
        :param valid_knee_angle: the x test input that we are getting in real time, shaped as (n_reps, 1, 1, window_length)
        :return: updates the reference trajectory continuously so we can just go check it
        """
        max_error_margin = int(2048 / (self.number_of_classes * window_step))
        print("The max error margin is", max_error_margin)
        start_updating = False
        if plot_continuously:
            fig = plt.figure()
            spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[5, 2], wspace=0.5, hspace=0)
            ax0 = fig.add_subplot(spec[0])  # This will be for the input trajectory
            ax1 = fig.add_subplot(spec[1])  # This is the current class text and predicted class text
            ax1.axis('off')
            ax2 = fig.add_subplot(spec[2])  # This will be for the reference trajectory
            ax3 = fig.add_subplot(spec[3])  # This is for the text that says recording or updating
            ax3.axis('off')
            plt.ion()
        with torch.no_grad():
            current_recorded_trajectories = []  # This is a list of all the trajectories corresponding to that same class
            next_class_trajectories = []
            for rep in range(valid_knee_angle.shape[0]):
                healthy_trajectory = valid_knee_angle[rep, :, :, :].to(device)
                predicted = torch.argmax(self.model(healthy_trajectory.float())).cpu().detach().item()
                if plot_continuously:
                    plt.show(block=False)
                    plt.pause(0.001)
                    if rep == 0:
                        plot0, = ax0.plot(healthy_trajectory.cpu().detach().numpy().squeeze())
                        plot2, = ax2.plot(self.segmented_reference_trajectory.reshape((-1, 1)).squeeze())
                    else:
                        plot0.set_ydata(healthy_trajectory.cpu().detach().numpy().squeeze())
                # CHECK IF THERE IS A CURRENT A CLASS AND INITIALISE IF NOT ============================================
                if self.current_class is None:
                    self.current_class = int(predicted)
                # CHECK IF THE PREDICTED CLASS IS THE SAME AS THE CURRENT CLASS ========================================
                if start_updating:
                    current_recorded_trajectories.append(
                        healthy_trajectory.cpu().detach().numpy().squeeze()[0:window_step])
                    if plot_continuously:
                        ax1.clear()
                        ax1.axis('off')
                        ax1.text(0, 0.8, "Current class: " + str(self.current_class))
                        if int(predicted) == self.current_class:
                            ax1.text(0, 0.2, "Predicted class: " + str(predicted), bbox=dict(facecolor='green', alpha=0.2))
                        else:
                            ax1.text(0, 0.2, "Predicted class: " + str(predicted), bbox=dict(facecolor='blue', alpha=0.2))
                        ax3.clear()
                        ax3.axis('off')
                        ax3.text(0, 0.5, "Recording")
                else:
                    if rep == 0 and plot_continuously:
                        ax3.text(0, 0.5, "Waiting")
                        ax3.axis('off')
                        ax1.text(0, 0.8, "Current class: " + str(self.current_class))
                        ax1.text(0, 0.2, "Predicted class: " + str(predicted))
                if int(predicted) == self.current_class:
                    next_class_trajectories = []
                else:
                    next_class_trajectories.append(healthy_trajectory.cpu().detach().numpy().squeeze()[0:window_step])
                if len(next_class_trajectories) > max_error_margin:
                    if start_updating:
                        current_recorded_trajectories = current_recorded_trajectories[0:
                                                                                      len(current_recorded_trajectories)
                                                                                      - max_error_margin]
                        if plot_continuously:
                            ax3.clear()
                            ax3.axis('off')
                            ax3.text(0, 0.5, "Updating")

                        # CONCATENATE SIGNAL SNIPPETS ======================================================================
                        reference_trajectory_segment = np.concatenate(np.array(current_recorded_trajectories), axis=0)
                        # INTERPOLATE LENGTH ===============================================================================
                        segment_x = np.linspace(0, self.segmented_reference_trajectory.shape[1],
                                                len(reference_trajectory_segment))
                        required_segment_x = np.linspace(0, self.segmented_reference_trajectory.shape[1],
                                                         self.segmented_reference_trajectory.shape[1])
                        reference_trajectory_segment = interp1d(segment_x,
                                                                reference_trajectory_segment.squeeze())(required_segment_x)

                        # MERGE WITH START VALUE ===========================================================================
                        if self.recorded_first_values[self.current_class] != 0:
                            start_margin = int(len(reference_trajectory_segment) / 10)  # change at most the first 10%
                            first_value = self.recorded_first_values[self.current_class]
                            last_value = reference_trajectory_segment[start_margin]
                            curve_function = np.poly1d(np.polyfit([0, start_margin], [first_value, last_value], 1))
                            reference_trajectory_segment[0:start_margin] = curve_function(np.arange(start_margin))
                        # SMOOTHING ========================================================================================
                        reference_trajectory_segment = savgol_filter(reference_trajectory_segment,
                                                                     int(len(reference_trajectory_segment) / 5) + 1,
                                                                     polyorder=3)
                        # UPDATE THE NEXT START VALUE ======================================================================
                        if self.current_class < self.number_of_classes-2:
                            self.recorded_first_values[self.current_class+1] = reference_trajectory_segment[-1]
                        else:
                            self.recorded_first_values[0] = reference_trajectory_segment[-1]
                        # UPDATE THE REFERENCE TRAJECTORY ==================================================================
                        self.segmented_reference_trajectory[self.current_class, :, 0] = reference_trajectory_segment
                        if plot_continuously:
                            plot2.set_ydata(self.segmented_reference_trajectory.reshape((-1, 1)).squeeze())
                    else:
                        start_updating = True
                    # UPDATE THE CURRENT CLASS AND CURRENT RECORDED TRAJECTORIES
                    # plt.pause(1)
                    if self.current_class == self.number_of_classes-1:
                        self.current_class = 0
                    else:
                        self.current_class = self.current_class + 1
                    current_recorded_trajectories = next_class_trajectories
                    next_class_trajectories = []


if __name__== "__main__":
    list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
    dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
    list_of_speeds = ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
    list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO",
                       "GM",
                       "GL"]
    list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]
    window_step = 10
    # trajectory_data = ReferenceTrajectoryData([list_of_subjects[0]], list_of_speeds, list_of_muscles, 'Knee', 'R',
    #                                           label_type='cycle percentage')
    classes = [5]
    for nclass in classes:
        trajectory_data = ReferenceTrajectoryData(['DS01'], list_of_speeds[0:-1], list_of_muscles, 'Knee',
                                                  window_length=500, label_type='percentage based classes',
                                                  number_of_classes=nclass, window_step=window_step)
        network = TrajectoryMappingNetwork(learning_rate=0.00001,
                                           epochs=500, saved_model_name='percentage_based_classes_500_' + str(nclass),
                                           model_type='classifier', window_length=500, number_of_classes=nclass, load=True)
        # network.test_network(trajectory_data.x_test, trajectory_data.y_test)
        reference_trajectory = ReferenceTrajectoryUpdate(initial_reference_trajectory=trajectory_data.reference_knee_angle,
                                                         initial_reference_label=trajectory_data.reference_label,
                                                         trained_model_name=network.saved_name, window_step=window_step,
                                                         model=network.model, number_of_classes=nclass)
        # reference_trajectory.plot_gait_cycle_segmentation()
        # plt.show()
        reference_trajectory.update_reference_trajectory_classification(valid_knee_angle=trajectory_data.x_test,
                                                                        window_step=window_step, plot_continuously=True)