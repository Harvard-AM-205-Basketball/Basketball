"""
Harvard IACS Applied Math 205
Project: Basketball

synchronize_audio.py
Synchronize the audio feeds from the different cameras

Michael S. Emanuel
Sat Dec 15 15:04:46 2018
"""

import numpy as np
from scipy.io.wavfile import read as read_wav
from scipy.signal import fftconvolve
# import matplotlib.pyplot as plt
from tqdm import tqdm
from am205_utils import range_inc

from typing import Tuple, List


def load_wave(path: str, camera_name: str) -> Tuple[int, np.ndarray]:
    """Loads one wave file from file path and name; returns one array of shape (M, ) of 8 bit integers"""
    # Generate the file name
    fname = path + '/' + f'{camera_name}.wav'
    # Read both channels of the audio
    rate: int
    data: np.ndarray
    rate, data = read_wav(fname)
    # These cameras record in mono; return the mean of both channels
    return np.mean(data, axis=1, dtype=np.float32)

   
def synchronize_streams(data_i: np.ndarray, data_j: np.ndarray):
    """
    Synchronize two audio streams.
    data_i: the first audio stream
    data_j: the second audio stream
    Returns:
    dt; this is the offset that reflects how much later data_j is than data_i.
    data_i[dt:dt+L] should overlap with data_j[0:L]    
    """
    # length of data_j
    L_j: int = len(data_j)

    # Compute the convolution between the two streams with the FFT
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve
    conv = fftconvolve(data_i, data_j[::-1])
    # The time offsset is the maximum of the convolution
    # dt = L_j - np.argmax(conv) - 1
    lag = np.argmax(conv) - L_j + 1
    return lag


def test_synchronize_streams():
    """Test synchronize function by applying a known lag to one audio stream and recovering it."""
    # Load one audio stream
    data_1: np.ndarray = load_wave(path_waves, 'Camera1.wav')
    
    # Create a second audio stream delayed by exactly 1.0 second
    lag_sec: float = 2.0
    lag_true: int = int(rate * lag_sec)
    data_2: np.ndarray = data_1[lag_true:]
    
    # Compute the lag; should recover rate
    lag_calc: int = synchronize_streams(data_1, data_2)
    
    # Tolerance for this test: resolution should be within 1 frame; there are 15 frames per second
    error_secs: float = abs(lag_calc - lag_true) / rate
    msg: str = 'PASS' if error_secs < (1.0 / 15) else 'FAIL'
    # Report results
    print(f'Testing audio synchronize function:')
    print(f'Applied a lag of {lag_sec} seconds / {lag_true} bins to data_1 to produce data_2.')
    print(f'Recovered lag = {lag_calc}.  Error = {error_secs:0.3f} seconds.')
    print(f'*** {msg} ***')

    
def audio_corr(data_i, data_j, lag) -> float:
    """Compute the correlation between two audio streams at the computed lag"""
    # Lag must be noon-negative; otherwise flip the order around
    if lag < 0:
        return audio_corr(data_j, data_i, -lag)
    # Length of overlapping data
    L: int = min(len(data_i)-lag, len(data_j))
    # Overlapping series
    x = data_i[lag:L+lag]
    y = data_j[0:L]
    # Return the correlation coefficient between these two series
    return np.corrcoef(x,y)[0,1]


def synchronize_cameras(camera_name_i: str, camera_name_j: str):
    """Synchronize two camera based on their audio streams."""
    # The audio streams for these cameras
    data_i = load_wave(path_waves, camera_name_i)
    data_j = load_wave(path_waves, camera_name_j)
    # Synchronize these audio streams
    return synchronize_streams(data_i, data_j)


def make_synch_matrix(camera_names):
    """Build an nxn matrix for joint estimation of synchronization"""
    # Get the number of cameras
    n: int = len(camera_names)
    # Initialize nxn matrix of lags
    lag_mat: np.ndarray = np.zeros((n,n), dtype=np.int32)
    # Initialize nxn matrix of recovered audio correlation
    corr_mat: np.ndarray = np.zeros((n,n))
    
    # Calibrate the lag for each pair of cameras
    for i, camera_name_i in enumerate(tqdm(camera_names)):
        data_i = load_wave(path_waves, camera_name_i)
        for j, camera_name_j in enumerate(camera_names):
            data_j = load_wave(path_waves, camera_name_j)
            # lag_mat[i, j] = synchronize_cameras(camera_name_i, camera_name_j)
            lag_ij = synchronize_streams(data_i, data_j)
            lag_mat[i, j] = lag_ij
            corr_mat[i, j] = audio_corr(data_i, data_j, lag_ij)

    # Return both the synchronization matrix and implied audio correlation matrix
    return lag_mat, corr_mat


# *************************************************************************************************
# Path to audio (WAV) directory
path_waves: str = r'../audio'

# Sampling rate
rate: int = 96000

# List of Camera names
# camera_names: List[str] = [f'Camera{n}' for n in range_inc(1, 8) if n != 5]
camera_names= ['Camera1', 'Camera3', 'Camera4']
# Number of cameras
camera_count: int = len(camera_names)

# Test synchronization
# test_synchronize_streams()

lag_mat, corr_mat = make_synch_matrix(camera_names)

lag_secs = np.round(lag_mat / rate, 2)
