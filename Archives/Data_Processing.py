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


def generate_key_event_label(time_series_label, filter=True):
    """
    This function will aim at generating classification labels for key events specified in the Koopman paper
    :param time_series_label: knee angle
    :return: a vector that is of the same length of the time-series, containing labels ranging from 0 to 5
    """
    if filter:
        time_series_label = savgol_filter(time_series_label, window_length=999, polyorder=2, axis=0)
    blues = ['#154360', '#1F618D', '#2471A3', '#5499C7', '#7FB3D5', '#D4E6F1']
    max_swing_peaks, _ = find_peaks(time_series_label.squeeze(), height=30, distance=2000)
    max_swing_peaks = np.insert(max_swing_peaks, 0, 0)
    max_swing_peaks = np.append(max_swing_peaks, len(time_series_label))

    # heel_strikes, _ = find_peaks(-time_series_label.squeeze(), height=-10, distance=2000)
    heel_strikes = []
    for i in range(len(max_swing_peaks)-1):
        # note that we start on a max swing peak, so we can use these to then find the other key events
        portion_of_interest = time_series_label[max_swing_peaks[i]:max_swing_peaks[i + 1]]
        heel_strike, _ = find_peaks(-portion_of_interest.squeeze())
        heel_strikes.append(heel_strike[0] + max_swing_peaks[i])

    recorded_local_max_stance = 0.33
    recorded_local_min_stance = 0.2
    max_stance = []
    min_stance = []
    for i in range(len(heel_strikes)):
        # note that in our dataset, all the heel strikes are followed by a max swing
        start, end = (heel_strikes[i], max_swing_peaks[i + 1])
        portion_of_interest = time_series_label[start:end, :]
        local_max_stance_peaks, _ = find_peaks(portion_of_interest.squeeze())
        local_max_stance_prominence = peak_prominences(portion_of_interest.squeeze(), local_max_stance_peaks)[0]
        if len(local_max_stance_peaks) == 0:
            local_max_stance_peak = int(len(portion_of_interest) * recorded_local_max_stance)
        else:
            local_max_stance_peak = local_max_stance_peaks[np.argmax(local_max_stance_prominence)]
        max_stance.append(int(local_max_stance_peak + start))
        start = start + local_max_stance_peak
        portion_of_interest = portion_of_interest[local_max_stance_peak::]
        local_min_stance_peaks, _ = find_peaks(-portion_of_interest.squeeze())
        local_min_stance_prominence = peak_prominences(-portion_of_interest.squeeze(), local_min_stance_peaks)[0]
        if len(local_min_stance_peaks) == 0:
            gradient = np.gradient(portion_of_interest, axis=0)
            min_gradient, _ = find_peaks(-gradient.squeeze())
            local_min_stance_peak = min_gradient[-1]
            # local_min_stance_peak = int(len(portion_of_interest) * recorded_local_min_stance)
        else:
            local_min_stance_peak = local_min_stance_peaks[-1]#[np.argmin(local_min_stance_prominence)]
        if local_min_stance_peak < 200:
            local_min_stance_peak = 200
        min_stance.append(int(local_min_stance_peak + start))

    mid_end_swing = []
    for i in range(len(max_swing_peaks) - 1):
        portion_of_interest = time_series_label[max_swing_peaks[i]:heel_strikes[i], :]
        mid_end_swing.append(int(len(portion_of_interest)/2+max_swing_peaks[i]))

    mid_start_swing = []
    for i in range(len(max_swing_peaks)-1):
        portion_of_interest = time_series_label[min_stance[i]:max_swing_peaks[i+1]:, :]
        mid_start_swing.append(int(len(portion_of_interest)/2 + min_stance[i]))

        # """ # Plot the classes with the labels as vertical lines
        # plt.plot(time_series_label)
        # for HS in heel_strikes:
        #     plt.axvline(HS, color='red')
        # for MAXST in max_stance:
        #     plt.axvline(MAXST, color=blues[1])
        # for MINST in min_stance:
        #     plt.axvline(MINST, color=blues[2])
        # for MSS in mid_start_swing:
        #     plt.axvline(MSS, color=blues[3])
        # for MAXS in max_swing_peaks:
        #     plt.axvline(MAXS, color=blues[4])
        # for MES in mid_end_swing:
        #     plt.axvline(MES, color=blues[5])
        # plt.show()
        # """

    """ # Plot the class indices to check for overlap
    plt.plot(heel_strikes, label='HS')
    plt.plot(max_stance, label='MaxSt')
    plt.plot(min_stance, label='MinSt')
    plt.plot(mid_start_swing, label='MSS')
    plt.plot(max_swing_peaks, label='MaxSP')
    plt.plot(mid_end_swing, label='MES')
    plt.legend()
    plt.show()
    exit()
    #"""

    # let's do the label as heel_strikes == 1, max_stance == 2, min_stance == 3, mid_start_swing == 4,
    # max_swing_peaks == 5, mid_end_swing == 6
    # print(len(heel_strikes), len(max_stance), len(min_stance), len(mid_start_swing), len(max_swing_peaks), len(mid_end_swing))
    max_swing_peaks[-1] = max_swing_peaks[-1] - 1
    label_vector = np.zeros_like(time_series_label)
    label_vector[heel_strikes, :] = 0
    label_vector[max_stance, :] = 1
    label_vector[min_stance, :] = 2
    label_vector[mid_start_swing, :] = 3
    label_vector[max_swing_peaks, :] = 4
    label_vector[mid_end_swing, :] = 5

    grouped = []
    group = []
    for i in range(len(label_vector)):
        if label_vector[i] == 0:
            group.append(i)
        else:
            if len(group) > 0:
                grouped.append(group)
                group = []

    for group in grouped:
        mid_point = group[int(len(group)/2)]
        label_vector[group[0]:mid_point, :] = label_vector[group[0]-1, :]
        label_vector[mid_point:group[-1]+1, :] = label_vector[group[-1]+1, :]

    return label_vector.astype(int)