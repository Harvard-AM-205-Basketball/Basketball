"""
Harvard IACS Applied Math 205
Project: Basketball

synchronize_audio.py
Synchronize the audio feeds from the different cameras

Michael S. Emanuel
Sat Dec 15 15:04:46 2018
"""

import numpy as np
from numpy.fft import fft
from scipy.io.wavfile import read as read_wav
from scipy.signal import convolve
import matplotlib.pyplot as plt
from tqdm import tqdm
from am205_utils import range_inc
from typing import Tuple, List


def load_wave_rate(path: str, camera_name: str) -> Tuple[int, np.ndarray]:
    """Loads one wave file from file path and name; the rate and one array of shape (M, ) of 16 bit integers"""
    # Generate the file name
    fname = path + '/' + f'{camera_name}.wav'
    # Read both channels of the audio
    rate: int
    data: np.ndarray
    rate, data = read_wav(fname)    
    # These cameras record in mono; compute the mean of both channels
    wav  = np.mean(data, axis=1, dtype=np.float32)
    # Return the rate and the wave
    return rate, wav


def load_wave(path: str, camera_name: str) -> np.ndarray:
    """Loads one wave file from file path and name; returns one array of shape (M, ) of 16 bit integers"""
    # Dispatch call to load_wave_rate
    rate, wav = load_wave_rate(path, camera_name)
    # Return just the wave
    return wav


def wav_to_freq(wav: np.ndarray, N: int):
    """Transform a wave input to sampled frequencies"""
    # https://engineersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
    # Number of windows
    K: int = len(wav) // N
    # Initialize KxN array with the fourier spectrum in seach window
    spectrum = np.zeros((K, N), dtype=np.float32)
    
    # Scale the wav to be in the interval -1, +1
    wav = wav + 2**15
    wav = wav / (2**16-1) * 2.0 - 1.0
    
    # Process each window
    for k in range(K):
        # Start and end index of this window
        i0: int = (k+0)*N
        i1: int = (k+1)*N
        # Extract the window
        x = wav[i0:i1]
        # Take FFT
        y = fft(x)
        # Single sided spectrum only
        y[1:] = 2*y[1:]
        # Get rid of imaginary part
        y = np.abs(y)
        # Save y to row k of the spectrum
        spectrum[k,:] = y

    # Return the Fourier spectrum
    return spectrum


def plot_spectrum_k(spectrum, k):
    """Plot the kth window from the spectrum"""
    fig,ax = plt.subplots(figsize=[12,8])
    plt.plot(freq, spectrum[k,:], linewidth=4)
    ax.set_xscale('log')
    plt.ylabel('log(Amplitude)')
    plt.xlabel('Frequency [Hz]')
    plt.show()


def synchronize_spectra(spec_i: np.ndarray, spec_j: np.ndarray, N: int) -> Tuple[int, float]:
    """
    Synchronize two audio streams given their fourier spectra.
    spec_i: the fourier spectrum of the first audio stream
    spec_j: the fourier spectrum of the second audio stream
    width: how wide to search (in seconds)
    Returns:
    dt; this is the offset that reflects how much later data_j is than data_i.
    data_i[dt:dt+L] should overlap with data_j[0:L]    
    """
    # Number of samples in the second spectrum
    L_j = len(spec_j)
    # Convolve the 2 spectra
    conv = convolve(spec_i, spec_j[::-1,:])
    # Take the square of the convolution
    conv2 = np.mean(conv*conv, axis=1)
    # Find the window offset that maximizes overlap
    # lag_spec = L_j - np.argmax(conv2)
    lag_spec = np.argmax(conv2) - L_j + 1
    # Window size in seconds
    window_size: float = N / rate
    # Lag in seconds
    lag_secs = lag_spec * window_size
    return lag_spec, lag_secs


def synchronize_streams(wav_i: np.ndarray, wav_j: np.ndarray, N: int) -> float:
    """
    Synchronize two audio streams.
    data_i: the first audio stream
    data_j: the second audio stream
    N:      sample window for frequency estimation
    Returns:
    dt; this is the offset that reflects how much later data_j is than data_i.
    data_i[dt:dt+L] should overlap with data_j[0:L]    
    """
    # Extract the spectra
    spec_i = wav_to_freq(wav_i, N)
    spec_j = wav_to_freq(wav_j, N)

    # Compute the convolution between the two streams with the FFT
    lag_spec, lag_secs = synchronize_spectra(spec_i, spec_j, N)
    # Convert the lag from number of windows into seconds
    return lag_secs


def test_synchronize_spectra(N: int):
    """Test synchronize function by applying a known lag to one audio stream and recovering it."""
    # Load one audio stream
    wav_1: np.ndarray = load_wave(path_waves, 'Camera1')
    
    # Create a second audio stream delayed by exactly 1.0 second
    lag_secs_true: float = 2.0
    lag_wav: int = int(rate * lag_secs_true)
    wav_2: np.ndarray = wav_1[lag_wav:]
    
    # Generate spectra from both signals
    spec_1 = wav_to_freq(wav_1, N)
    spec_2 = wav_to_freq(wav_2, N)
    
    # Compute the lag; should recover rate
    lag_spec: int
    lag_secs_calc: float
    lag_spec, lag_secs_calc = synchronize_spectra(spec_1, spec_2, N)
    
    # Tolerance for this test: resolution should be within 1 frame; there are 15 frames per second
    error_secs: float = abs(lag_secs_calc - lag_secs_true)
    msg: str = 'PASS' if error_secs < (1.0 / 15) else 'FAIL'
    # Report results
    print(f'Testing audio synchronize function:')
    print(f'Applied a lag of {lag_secs_true} seconds / {lag_wav} bins to data_1 to produce data_2.')
    print(f'Recovered lag = {lag_secs_calc:0.3f}.  Error = {error_secs:0.3f} seconds.')
    print(f'*** {msg} ***')


def audio_corr(spec_i, spec_j, lag) -> float:
    """Compute the correlation between two audio spectra at the computed lag"""
    # Lag must be noon-negative; otherwise flip the order around
    if lag < 0:
        return audio_corr(spec_j, spec_i, -lag)
    # Length of overlapping data
    L: int = min(len(spec_i)-lag, len(spec_j))
    # Overlapping series
    x = spec_i[lag:L+lag, :]
    y = spec_j[0:L, :]
    # Comput
    xx = np.sum(x * x)
    yy = np.sum(y * y)
    xy = np.sum(x * y)
    # Return the correlation coefficient between these two series
    return xy / np.sqrt(xx * yy)


def make_synch_matrix(spectra, N: int):
    """Build an nxn matrix for joint estimation of synchronization"""
    # Get the number of cameras
    n: int = len(spectra)
    # Initialize nxn matrix of lags
    # lag_mat_i: np.ndarray = np.zeros((n, n), dtype=np.int32)
    lag_mat: np.ndarray = np.zeros((n,n), dtype=np.int32)
    # Initialize nxn matrix of recovered audio correlation
    corr_mat: np.ndarray = np.zeros((n, n))
    # Length of each window
    window_secs: float = N / rate
    
    # Calibrate the lag for each pair of cameras
    for i, spec_i in enumerate(tqdm(spectra)):
        for j, spec_j in enumerate(spectra):
            lag_ij = synchronize_spectra(spec_i, spec_j)
            # lag_mat_i[i, j] = lag_ij
            lag_mat[i, j] = lag_ij * window_secs
            corr_mat[i, j] = audio_corr(spec_i, spec_j, lag_ij)

    # Return both the synchronization matrix and implied audio correlation matrix
    return lag_mat, corr_mat


def test_synch_matrix():
    """Test synchronize function by applying a known lag to one audio stream and recovering it."""
    # Load one audio stream
    wav_1: np.ndarray = load_wave(path_waves, 'Camera1')
    
    # Set window size
    N: int = 4096
    # Length of each window
    window_secs: float = N / rate
    
    # Lags to apply to cameras 2 and 3
    lag_2_sec: float = 1.0
    lag_3_sec: float = 3.0
    
    # Lags in wave space
    lag_2_wav = round(lag_2_sec * rate)
    lag_3_wav = round(lag_3_sec * rate)
    
    # Create a two audio streams delayed by exactly 1.0 and 3.0 seconds
    wav_2: np.ndarray = wav_1[lag_2_wav:]
    wav_3: np.ndarray = wav_1[lag_3_wav:]

    # Waves and spectra for the three example
    wavs = (wav_1, wav_2, wav_3)
    spectra = [wav_to_freq(wav,N) for wav in wavs]
    
    # Initialize arrays for the computed lags and correlations
    lag_secs = np.zeros((3,3))
    corr_mat = np.zeros((3,3))
    
    # Estimate all 9 lags and correlations
    for i in range(3):
        spec_i = spectra[i]
        for j in range(3):
            spec_j = spectra[j]
            lag_ij_spec, lag_ij_secs = synchronize_spectra(spec_i, spec_j, N)
            lag_secs[i, j] = lag_ij_secs
            corr_mat[i, j] = audio_corr(spec_i, spec_j, lag_ij_spec)
        
    # Known correct answer
    lag_secs_true = np.array(
           [[0.0, 1.0, 3.0], 
            [-1.0, 0.0, 2.0],
            [-3.0, -2.0, 0.0]])
    # Errors
    err_mat = lag_secs - lag_secs_true
    max_err = np.max(np.abs(err_mat))

    # Report resultss
    is_ok = max_err < window_secs
    msg = 'PASS' if is_ok else 'FAIL'
    print(f'\nTesting make_synch_matrix: maximum error = {max_err:0.3f}')
    print('Recovered Lag Matrix:')
    print(lag_secs)
    print('Correlation Matrix:')
    print(corr_mat)
    print(f'*** {msg} ***')
    return lag_mat, corr_mat


# *************************************************************************************************
# Path to audio (WAV) directory
path_waves: str = r'../audio'

# Sampling rate
rate: int = 48000

# List of Camera names
# camera_names: List[str] = [f'Camera{n}' for n in range_inc(1, 8) if n != 5]
camera_names = [f'Camera{n}' for n in range_inc(1, 3)]

# Number of cameras
camera_count: int = len(camera_names)

# Sample window for Fourier spectrum
N: int = 4096
# window_secs
window_secs: float = N / rate

# Frequency vector for plots
freq = rate*np.arange((N))/(2*N);

# Waves and spectra for all inputs
waves = list()
spectra = list()
for i, camera_name in enumerate(camera_names):
    # Get the sampling rate and wave form of camera i
    rate_i, wav_i = load_wave_rate(path_waves, camera_name)
    # Check that the sampling rate matches
    assert rate_i == rate, 'Error: {camera_name} has sample rate {rate_i} != {rate}!'
    # Save the wave form
    waves.append(wav_i)
    # Extract the Fourier spectrum for this wave
    spectra.append(wav_to_freq(waves[i], N))

# Test synchronization
test_synchronize_spectra(N)

# Test the synchronization matrix generation routine
lag_secs, corr_mat = test_synch_matrix()


# Generate the matrix of of lags and correlations on the real data
# lag_mat, corr_mat = make_synch_matrix(spectra)

