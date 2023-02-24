import math
import numpy as np
from torch.nn.functional import one_hot
import c3d
from scipy.signal import find_peaks, peak_prominences
import pandas as pd
import matplotlib.pyplot as plt
import torch
from itertools import groupby
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


# GENERAL DATA EXTRACTION FUNCTIONS ====================================================================================
def extract_hdf5_data_to_EMG_and_labels(subject, speed, list_of_muscles, dominant_leg, joint="Knee"):
    """
    Get the Delsys data from the storage files to be able to use. In this case, we are extracting the data one speed at
    a time. Note that we only want to use subjects DS01, DS02, DS04, DS05, DS06, DS07, and that the speeds 07 and 08 of
    the DS07 are corrupt, so only use speeds 09 onwards
    :param subject: the name of the subject we want to extract data from
    :param speed: the steady-state speed we are extracting
    :param list_of_muscles: the list of muscles we want to get
    :param label_name: the angular joint label
    :return: 3 arrays, one containing the EMG signals, another containing the joint angles of the paretic leg, and the
    last one containing the joint angles of the valid leg
    """
    data = pd.read_hdf('/media/ag6016/Storage/MuscleSelection/full_data/' + subject + '_' + speed + '.h5', key='Data')
    EMG_signals = data[list_of_muscles].to_numpy()
    dominant_label = np.expand_dims(data[str(dominant_leg + joint + "Angles")].to_numpy(), axis=1)
    if dominant_leg == 'R':
        valid_leg = 'L'
    elif dominant_leg == 'L':
        valid_leg = 'R'
    else:
        raise ValueError("The dominant leg can only be 'R' or 'L'")
    valid_label = np.expand_dims(data[str(valid_leg + joint + "Angles")].to_numpy(), axis=1)
    # crop signal at first and last peak
    peaks, _ = find_peaks(valid_label.squeeze(), height=30, distance=2000)
    start, end = (peaks[0], peaks[-1])
    EMG_signals = EMG_signals[start:end, :]
    dominant_label = dominant_label[start:end, :]
    valid_label = valid_label[start:end, :]
    # Because of the shift caused by the placement of the markers, we can assume that the minimum knee angle is 0,
    # so let's shift the knee angles of both legs so that the minimum is 0
    dominant_label = dominant_label - min(dominant_label)
    valid_label = valid_label - min(valid_label)
    return EMG_signals, dominant_label, valid_label


def generate_gait_cycle_percentage(time_series_label):
    """
    In this function, we want to generate a gait cycle percentage label that corresponds to the given knee angle data
    :param time_series_label: 1-D knee angle data
    :return: 1-D gait cycle percentage label
    """
    peaks, _ = find_peaks(time_series_label.squeeze(), height=30, distance=2000)
    # We add the indices of the first and last points
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, len(time_series_label))
    gait_cycle_percentages = []
    for peak in range(len(peaks)-1):
        cycle_length = peaks[peak+1] - peaks[peak]
        cycle_percentage = np.linspace(start=0, stop=100, num=cycle_length)
        gait_cycle_percentages.append(cycle_percentage)
    gait_cycle_percentages = np.expand_dims(np.concatenate(gait_cycle_percentages, axis=0), axis=1)
    return gait_cycle_percentages


def generate_percentage_based_classes(time_series_label, number_of_classes):
    gait_cycle_percentages = generate_gait_cycle_percentage(time_series_label)
    class_thresholds = np.linspace(0, 100, num=number_of_classes+1)
    print(class_thresholds)
    percentage_based_classes = []
    for i in range(len(gait_cycle_percentages)):
        if gait_cycle_percentages[i] == class_thresholds[0]:
            percentage_based_classes.append(int(0))
        elif gait_cycle_percentages[i] == 100:
            percentage_based_classes.append(int(number_of_classes-1))
        else:
            percentage_based_classes.append(int(np.where(class_thresholds > gait_cycle_percentages[i])[0][0]-1))
    return np.array(percentage_based_classes).astype(int)


def generate_reference_cycle(time_series_label, label_type, cycle_length=1000, number_of_classes=None):
    peaks, _ = find_peaks(time_series_label.squeeze(), height=30, distance=2000)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, len(time_series_label))

    all_cycles = []
    for i in range(1, len(peaks)):
        cycle = time_series_label[peaks[i-1]: peaks[i], :]
        old_x = np.linspace(0, cycle_length, len(cycle))
        new_x = np.linspace(0, cycle_length, cycle_length)
        f = interp1d(old_x, cycle.squeeze())
        cycle = f(new_x)
        all_cycles.append(np.expand_dims(cycle, axis=1))
    all_cycles = np.concatenate(all_cycles, axis=1)
    average_cycle = np.expand_dims(np.mean(all_cycles, axis=1), axis=1)
    if label_type == 'cycle percentage':
        average_label = generate_gait_cycle_percentage(average_cycle)
    elif label_type == 'percentage based classes':
        average_label = generate_percentage_based_classes(average_cycle, number_of_classes)
    else:
        raise ValueError('Wrong label type')

    return average_cycle, average_label


def split_data_into_TCN_windows(input_data, output_data, window_length=250, window_step=40, output_type='mapping'):
    """
    This function will split the data into windowed signals. The number of windows is always at the start
    :param input_data: time series array of shape (n_samples, n_channels)
    :param output_data: time series array of shape (n_samples, n_channels)
    :param window_length: integer for the number of samples we are looking at
    :param window_step: step between the extracted windows
    :param output_type: can either be "mapping" or "prediction", depending on whether we are doing a mapping or not
    :param reps_first: so that the output gives us the reps at the start
    :return: the windowed input and output signals, of shape (n_reps, n_channels, window_length) and (n_reps, n_channels, n_classes)
    """
    input_windows = []
    output_windows = []
    for i in range(0, int(input_data.shape[0]-int(window_length)), int(window_step)):
        input_windows.append(np.expand_dims(input_data[i:i+int(window_length), :], axis=0))
        if output_type == 'mapping':
            output_windows.append(np.expand_dims(output_data[i + int(window_length/2), :], axis=0))
        elif output_type == 'prediction':
            output_windows.append(np.expand_dims(output_data[i + int(window_length) + 1, :], axis=0))
    windowed_inputs = np.concatenate(input_windows, axis=0).transpose((0, 2, 1))
    windowed_outputs = np.expand_dims(np.concatenate(output_windows, axis=0), axis=-1).transpose((0, 2, 1))
    return windowed_inputs, windowed_outputs


def split_into_batches(input_data, output_data, batch_size):
    """
    This will take in the data and split it along the first axis
    :param input_data: time series of shape (n_reps, n_channels, window_length)
    :param output_data: time series of shape (n_reps, n_channels, 1)
    :param batch_size: the batch size
    :return: batched input and output data
    """
    # First let's snip the data so that it fits in the number of batches
    cropped_length = int(input_data.shape[0] - (input_data.shape[0] % batch_size))
    input_data = input_data[0:cropped_length, :, :]
    output_data = output_data[0:cropped_length, :, :]
    input_data = np.reshape(input_data, (-1, batch_size, input_data.shape[1], input_data.shape[2]), order='C')
    output_data = np.reshape(output_data, (-1, batch_size, output_data.shape[1], output_data.shape[2]), order='C')

    return input_data, output_data


# DATA EXTRACTION CLASSES ==============================================================================================
class ReferenceTrajectoryData:
    def __init__(self, subjects, list_of_speeds, list_of_muscles, joint, window_length=500, test_subject=None, batch_size=32,
                 label_type='cycle percentage', number_of_classes=None, window_step=40):
        """
        This class will aim to prepare and extract the data necessary for the first model that creates the reference
        trajectory.
        :param subjects: LIST of subjects (always input as a list)
        :param list_of_speeds: list of speeds
        :param list_of_muscles: list of muscles
        :param joint: name of the joint we are looking at as a string
        :param dominant_leg: as a string
        :param test_subject: as a string
        """
        self.test_subject = test_subject
        self.label_type = label_type
        self.window_length = window_length
        if test_subject is not None:
            # Enable the separation of a subject from the rest in case we want to do cross-subject testing
            self.subjects = [subject for subject in subjects if subject != test_subject]
        else:
            # Assume that we always have a list of subjects
            self.subjects = subjects
        self.list_of_speeds = list_of_speeds
        self.list_of_muscles = list_of_muscles
        self.joint = joint
        self.dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
        self.batch_size = batch_size
        # self.valid_leg = ['R' if dominant_leg == 'L' else 'L']
        self.number_of_classes = number_of_classes
        self.window_step = window_step
        self.valid_knee_angle = None
        self.label = None
        self.test_knee_angle = None
        self.test_label = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.reference_knee_angle = None
        self.reference_label = None
        # THIS IS FOR THE EMG DATA =====================================================================================
        self.paretic_knee_angle = None
        self.EMG_data = None
        self.test_paretic_knee_angle = None
        self.test_EMG_data = None
        # For the EMG data of the paretic leg
        self.EMG_train = None
        self.EMG_val = None
        self.EMG_test = None
        # For the trajectory data of the paretic leg
        self.paretic_train = None
        self.paretic_val = None
        self.paretic_test = None

        self.extract_data()
        self.get_reference_trajectory()
        self.format_data_for_TCN_with_batches()
        self.convert_data_into_tensors()

    def extract_data(self):
        all_trajectory_labels = []
        all_emg_data = []
        all_paretic_labels = []
        for subject in self.subjects:
            paretic_leg = self.dominant_leg[subject]
            # valid_leg = ['R' if paretic_leg == 'L' else 'L'][0]
            for speed in self.list_of_speeds:
                EMG_data, paretic_knee_angle, valid_knee_angle = \
                    extract_hdf5_data_to_EMG_and_labels(subject=subject, speed=speed,
                                                        list_of_muscles=self.list_of_muscles, dominant_leg=paretic_leg,
                                                        joint=self.joint)
                all_emg_data.append(EMG_data)
                all_paretic_labels.append(paretic_knee_angle)
                all_trajectory_labels.append(valid_knee_angle)
        self.EMG_data = np.concatenate(all_emg_data, axis=0)
        self.paretic_knee_angle = np.concatenate(all_paretic_labels, axis=0)
        self.valid_knee_angle = np.concatenate(all_trajectory_labels, axis=0)
        if self.label_type == 'cycle percentage':
            self.label = generate_gait_cycle_percentage(self.valid_knee_angle)
        elif self.label_type == 'percentage based classes':
            self.label = generate_percentage_based_classes(self.valid_knee_angle, self.number_of_classes)
            self.label = np.eye(self.number_of_classes)[self.label].squeeze()
        else:
            raise ValueError("Label type can only be cycle percentage or key events or percentage based classes")
        if self.test_subject is not None:
            test_paretic_leg = self.dominant_leg[self.test_subject]
            # test_valid_leg = ['R' if test_paretic_leg == 'L' else 'L'][0]
            all_trajectory_labels = []
            all_emg_data = []
            all_paretic_labels = []
            for speed in self.list_of_speeds:
                EMG_data, paretic_knee_angle, valid_knee_angle = \
                    extract_hdf5_data_to_EMG_and_labels(subject=self.test_subject, speed=speed,
                                                        list_of_muscles=self.list_of_muscles,
                                                        dominant_leg=test_paretic_leg, joint=self.joint)
                all_emg_data.append(EMG_data)
                all_paretic_labels.append(paretic_knee_angle)
                all_trajectory_labels.append(valid_knee_angle)
            self.test_EMG_data = np.concatenate(all_emg_data, axis=0)
            self.test_paretic_knee_angle = np.concatenate(all_paretic_labels, axis=0)
            self.test_knee_angle = np.concatenate(all_trajectory_labels, axis=0)
            if self.label_type == 'cycle percentage':
                self.test_label = generate_gait_cycle_percentage(self.test_knee_angle)
            elif self.label_type == 'percentage based classes':
                self.test_label = generate_percentage_based_classes(self.test_knee_angle, self.number_of_classes)
            else:
                raise ValueError("Label type can only be cycle percentage or key events or percentage based classes")

    def format_data_for_TCN_with_batches(self):
        """
        Format the data into batched train, validation and testing sets.
        :return: All the x have the shape (n_batches, batch_size, n_channels, window_length) and y have the shape
        (n_batches, batch_size, 1, 1) for the training, and then (n_reps, n_channels, window_length) and (n_reps, 1, 1)
        for testing
        """

        self.valid_knee_angle, self.label = split_data_into_TCN_windows(self.valid_knee_angle, self.label,
                                                                        window_length=self.window_length,
                                                                        window_step=self.window_step,
                                                                        output_type='mapping')

        self.EMG_data, self.paretic_knee_angle = split_data_into_TCN_windows(self.EMG_data, self.paretic_knee_angle,
                                                                             window_length=self.window_length,
                                                                             window_step=self.window_step,
                                                                             output_type='prediction')
        if self.test_subject is not None:
            # if we have a reserve test subject, then we can just shuffle all the other subjects together
            self.x_test, self.y_test = split_data_into_TCN_windows(self.test_knee_angle, self.test_label,
                                                                   output_type='mapping')
            self.EMG_test, self.paretic_test = split_data_into_TCN_windows(self.test_EMG_data,
                                                                           self.test_paretic_knee_angle,
                                                                           output_type='prediction')
            # SPLIT INTO BATCHES
            self.valid_knee_angle, self.label = split_into_batches(self.valid_knee_angle, self.label, self.batch_size)
            self.EMG_data, self.paretic_knee_angle = split_into_batches(self.EMG_data, self.paretic_knee_angle,
                                                                        self.batch_size)
            # SPLIT INTO TRAIN VAL
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.valid_knee_angle, self.label,
                                                                                  train_size=0.8, random_state=42,
                                                                                  shuffle=True)
            self.EMG_train, self.EMG_val, self.paretic_train, self.paretic_val = train_test_split(self.EMG_data,
                                                                                                  self.paretic_knee_angle,
                                                                                                  train_size=0.8,
                                                                                                  random_state=42,
                                                                                                  shuffle=True)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.valid_knee_angle, self.label,
                                                                                    train_size=0.9, shuffle=False)
            self.EMG_train, self.EMG_test, self.paretic_train, self.paretic_test = train_test_split(self.EMG_data,
                                                                                                    self.paretic_knee_angle,
                                                                                                    train_size=0.9,
                                                                                                    shuffle=False)
            # SPLIT INTO BATCHES
            self.x_train, self.y_train = split_into_batches(self.x_train, self.y_train, self.batch_size)
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                                  train_size=0.9, random_state=42,
                                                                                  shuffle=True)
            self.EMG_train, self.paretic_train = split_into_batches(self.EMG_train, self.paretic_train, self.batch_size)
            self.EMG_train, self.EMG_val, self.paretic_train, self.paretic_val = train_test_split(self.EMG_train,
                                                                                                  self.paretic_train,
                                                                                                  train_size=0.9,
                                                                                                  random_state=42,
                                                                                                  shuffle=True)
        self.x_test = np.expand_dims(self.x_test, axis=1)
        self.y_test = np.expand_dims(self.y_test, axis=1)
        self.EMG_test = np.expand_dims(self.EMG_test, axis=1)
        self.paretic_test = np.expand_dims(self.paretic_test, axis=1)

    def convert_data_into_tensors(self):
        self.x_train = torch.autograd.Variable(torch.from_numpy(self.x_train), requires_grad=False)
        self.y_train = torch.autograd.Variable(torch.from_numpy(self.y_train), requires_grad=False)
        self.x_val = torch.autograd.Variable(torch.from_numpy(self.x_val), requires_grad=False)
        self.y_val = torch.autograd.Variable(torch.from_numpy(self.y_val), requires_grad=False)
        self.EMG_train = torch.autograd.Variable(torch.from_numpy(self.EMG_train), requires_grad=False)
        self.paretic_train = torch.autograd.Variable(torch.from_numpy(self.paretic_train), requires_grad=False)
        self.EMG_val = torch.autograd.Variable(torch.from_numpy(self.EMG_val), requires_grad=False)
        self.paretic_val = torch.autograd.Variable(torch.from_numpy(self.paretic_val), requires_grad=False)
        self.x_test = torch.from_numpy(self.x_test)
        self.y_test = torch.from_numpy(self.y_test)
        self.EMG_test = torch.from_numpy(self.EMG_test)
        self.paretic_test = torch.from_numpy(self.paretic_test)

    def get_reference_trajectory(self):
        self.reference_knee_angle, self.reference_label = generate_reference_cycle(self.valid_knee_angle,
                                                                                   self.label_type,
                                                                                   number_of_classes=self.number_of_classes)


if __name__ == "__main__":
    list_of_subjects = ['DS01', 'DS02', 'DS04', 'DS05', 'DS06', 'DS07']
    dominant_leg = {'DS01': 'R', 'DS02': 'L', 'DS04': 'L', 'DS05': 'R', 'DS06': 'R', 'DS07': 'L'}
    list_of_speeds = ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', 'R']
    list_of_muscles = ["ESL-L", "ESL-R", "ESI-L", "ESI-R", "MF-L", "MF-R", "RF", "VM", "VL", "BF", "ST", "TA", "SO",
                       "GM",
                       "GL"]
    list_joint_angles = ["HipAngles", "KneeAngles", "AnkleAngles"]

    trajectory_data = ReferenceTrajectoryData(['DS01'], list_of_speeds, list_of_muscles, 'Knee',
                                              label_type='percentage based classes', number_of_classes=10)

    print("THIS IS FOR THE TRAJECTORY GENERATION")
    print(trajectory_data.x_train.shape)
    print(trajectory_data.y_train.shape)
    print(trajectory_data.x_test.shape)
    print(trajectory_data.y_test.shape)
    print("THIS IS FOR THE GAIT PREDICTION")
    print(trajectory_data.EMG_train.shape)
    print(trajectory_data.paretic_train.shape)
    print(trajectory_data.EMG_test.shape)
    print(trajectory_data.paretic_test.shape)

