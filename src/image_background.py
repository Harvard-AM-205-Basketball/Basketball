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
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from IPython.display import display
from am205_utils import range_inc
from typing import List

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


def load_frames(path_frames: str, camera_name: str) -> np.ndarray:
    """Load frame images; return a Txmxn array of 8 bit integers"""
    # Find all the frames in this directory
    fnames: List[str] = frame_names(path_frames, camera_name)
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'
    # Count the number of frames, T
    T: int = len(fnames)
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
    print(f'Importing {T} frames from {path}...')
    for T, fname in enumerate(tqdm(fnames)):
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
    

# *************************************************************************************************
def main():
    # Path to frames directory
    path_frames: str = r'../frames'
    # Path to directory of background frames
    path_background: str = r'../frames/Background'
    # List of Camera names
    camera_names: List[str] = [f'Camera{n}' for n in range_inc(8)]
    # Figure size for displaying mean and median
    figsize=[19.2, 10.8]

    # Iterate over all the cameras
    for camera_name in camera_names[0:2]:
        # Path with frames for this camera
        path: str = f'{path_frames}/{camera_name}'
        # Get frames for this camera
        frames = load_frames(path_frames, camera_name)
        
        # For each pixel, compute the mean
        mean_frame = calc_mean_frame(frames)
        # Display the mean frame
        print('Mean Frame')
        fig = plt.figure(figsize=figsize)
        io.imshow(mean_frame)
        display(fig)
        plt.close(fig)
        # Save the mean frame
        io.imsave(f'{path}/{camera_name}_mean.png', mean_frame)
        io.imsave(f'{path_background}/{camera_name}_mean.png', mean_frame)
        
        # For each pixel, compute the median
        median_frame = calc_median_frame(frames)
        # Display the median frame
        print('Median Frame')
        fig = plt.figure(figsize=figsize)
        io.imshow(median_frame)        
        display(fig)
        plt.close(fig)
        # Save the median frame
        io.imsave(f'{path}/{camera_name}_median.png', median_frame)
        io.imsave(f'{path_background}/{camera_name}_median.png', median_frame)

if __name__ == '__main__':
    main()
