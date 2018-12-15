"""
Harvard IACS Applied Math 205
Project: Basketball

image_background.py
Extract the background from a sequence of frames taken by the same camera.
Background is inferred as either the mean or median value for each pixel.

Michael S. Emanuel
Sat Dec 15 08:48:28 2018
"""

import numpy as np
from skimage import io
from IPython.display import display
import matplotlib.pyplot as plt
import os
import re
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from am205_utils import range_inc
from typing import List, Optional

# *************************************************************************************************
# silence irrelevant user warnings
warnings.filterwarnings("ignore", category=UserWarning)


# *************************************************************************************************
def load_frame_i(path: str, fname: str) -> np.ndarray:
    """Loads one frame from file path and name; returns one array of shape (M, N, K) of 8 bit integers"""
    # Read the file as a 3D array with integer pixel values in range [0, 255]
    fname = path + '/' + fname
    RGB_i: np.array = io.imread(fname)
    return RGB_i


def load_frame(path: str, fname: str) -> np.ndarray:
    """Loads one frame from file path and name; returns one array of shape (M, N, K) of 32 bit floats"""
    # Load frame as 8 bit integers
    RGB_i = load_frame_i(path, fname)
    # Convert to an array of floats
    RGB: np.array = np.zeros_like(RGB_i, dtype=np.float32)
    RGB[:,:,:] = RGB_i / 255.0


def frame_names(path_frames: str, camera_name: str) -> List[str]:
    """Return a list of file names for frames in this directory"""
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'
    # Files in this path; frames are named e.g. 'Camera1_Frame01234.png'
    fnames: List[str] = os.listdir(path)
    # Filter the list of frames to only those matching the pattern
    pattern: re.Pattern = re.compile(f'^{camera_name}_Frame'+'(\d{5}).png$')
    # Return list of all file names matching this pattern
    return [fname for fname in fnames if pattern.match(fname) is not None]


def load_frames(path_frames: str, camera_name: str, max_frames: Optional[int]=None) -> np.ndarray:
    """Load frame images; return a Txmxn array of 8 bit integers"""
    # Find all the frames in this directory
    fnames: List[str] = frame_names(path_frames, camera_name)
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'

    # Count the number of frames, T
    T: int = len(fnames)
    # Limit the frames to max_frames if it was specified
    if max_frames is not None:
        T = min(T, max_frames)

    # Describe the shape of the data
    # M: number of rows (1080)
    M: int
    # N: number of columns (1920)
    N: int
    # K: number of color channels (3)
    K: int
    # Get the shape of the first image; MxNxK
    M, N, K = load_frame_i(path, fnames[0]).shape

    # Create a big array to store all the frames; shape (T, M, N, K)
    frames = np.zeros((T, M, N, K), dtype=np.uint8)
    # Iterate over all the frames in the directory for this camera
    # print(f'Importing {T} frames from {path}...')    
    # Create different iterators for Camera1 and the rest so we have
    # just one progress bar (tqdm gets messed up with progress bars on multiple threads)
    fname_iter = fnames[0:T]
    if camera_name == 'Camera1':
        fname_iter = tqdm(fname_iter)
        print(f'Progress Bar for Camera1:')
    # Iterate over the frames for this camera
    for T, fname in enumerate(fname_iter):
        # Load the RGB image for this frame into the slice for frame T
        frames[T,:,:,:] = load_frame_i(path, fname)
    # Return the frames
    return frames


def calc_mean_frame(frames) -> np.ndarray:
    """Compute the mean frame from an array of frames in 8 bit integer format"""
    # For each pixel, compute the mean and median
    return np.mean(frames, axis=0, dtype=np.float32) / 255.0


def calc_median_frame(frames) -> np.ndarray:
    """Compute the median frame from an array of frames in 8 bit integer format"""
    # For each pixel, compute the mean and median
    return np.median(frames, axis=0) / 255.0
    

def process_one_camera(path_frames, camera_name, max_frames: Optional[int]=None):
    """Process frames for one camera; inside of for loop for parallelization"""
    # Get frames for this camera
    frames = load_frames(path_frames, camera_name, max_frames)
   
    # For each pixel, compute the mean
    mean_frame = calc_mean_frame(frames)
    
    # For each pixel, compute the median
    median_frame = calc_median_frame(frames)
    
    return mean_frame, median_frame


# *************************************************************************************************
def main():
    # Path to frames directory
    path_frames: str = r'../frames'
    # Path to directory of background frames
    path_background: str = r'../frames/Background'
    # List of Camera names
    camera_names: List[str] = [f'Camera{n}' for n in range_inc(8) if n != 5]
    # Number of cameras
    camera_count: int = len(camera_names)
    
    # Figure size for displaying mean and median
    figsize=[19.2, 10.8]
    
    # Compute mean and median frame in parallel
    # https://joblib.readthedocs.io/en/latest/parallel.html
    print(f'Running {camera_count} parallel jobs on threads (1 for each camera)...')
    background = Parallel(n_jobs=camera_count, prefer='threads')(
        delayed(process_one_camera)(path_frames, camera_name)
        for camera_name in camera_names)
    
    # Save and display the background frames (mean and median)
    for i, camera_name in enumerate(camera_names):
        # Look up the results for this camera from parallel job
        mean_frame, median_frame = background[i]
    
        # Path with frames for this camera
        path: str = f'{path_frames}/{camera_name}'
    
        # Display and save the mean frame
        print(f'Mean Frame for {camera_name}')
        fig = plt.figure(figsize=figsize)
        io.imshow(mean_frame)
        display(fig)
        plt.close(fig)
        # Save the mean frame
        io.imsave(f'{path}/{camera_name}_mean.png', mean_frame)
        io.imsave(f'{path_background}/{camera_name}_mean.png', mean_frame)
        # Save mean frame as a numpy matrix
        np.save(f'{path_background}/{camera_name}_mean.npy', mean_frame)
        
        # Display and save the median frame
        print(f'Median Frame for {camera_name}')
        fig = plt.figure(figsize=figsize)
        io.imshow(median_frame)        
        display(fig)
        plt.close(fig)
        # Save the median frame
        io.imsave(f'{path}/{camera_name}_median.png', median_frame)
        io.imsave(f'{path_background}/{camera_name}_median.png', median_frame)
        np.save(f'{path_background}/{camera_name}_median.npy', median_frame)

if __name__ == '__main__':
    main()
