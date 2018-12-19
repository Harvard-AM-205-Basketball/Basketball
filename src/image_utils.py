"""
Harvard IACS Applied Math 205
Project: Basketball

image_utils.py
Utilities for working with frames on this project.

Michael S. Emanuel
Wed Dec 19 00:29:33 2018
"""

import os
import re
import numpy as np
from skimage import io
import tqdm
from typing import List, Optional


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
    return RGB


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
        print(f'\nProgress Bar for Camera1:')
    # Iterate over the frames for this camera
    for T, fname in enumerate(fname_iter):
        # Load the RGB image for this frame into the slice for frame T
        frames[T,:,:,:] = load_frame_i(path, fname)
    # Return the frames
    return frames
