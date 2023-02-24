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