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
import matplotlib.pyplot as plt
from skimage import io
from IPython.display import display
import tqdm
from typing import List, Optional

# *************************************************************************************************
# global variables
# Path with synchronized frames
path_frames = r'../sync_frames'


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


# *************************************************************************************************
def combine_frames(frames):
    """Combine an array of 6 frames into one giant frame for quick visualization"""
    combined_frame = np.zeros((3*1080, 2*1920, 3))
    for i in range(3):
        for j in range(2):
            i0 = 1080*i
            j0 = 1920*j
            combined_frame[i0:i0+1080, j0:j0+1920] = frames[2*i+j]
    return combined_frame


def make_tableau(n: int):
    """Make a "tableau" of synchronized frames at this frame number"""    
    # Load the frames
    frame2 = load_frame(f'{path_frames}/Camera2', f'Camera2_SyncFrame{n:05d}.png')
    frame3 = load_frame(f'{path_frames}/Camera3', f'Camera3_SyncFrame{n:05d}.png')
    frame4 = load_frame(f'{path_frames}/Camera4', f'Camera4_SyncFrame{n:05d}.png')
    frame6 = load_frame(f'{path_frames}/Camera6', f'Camera6_SyncFrame{n:05d}.png')
    frame7 = load_frame(f'{path_frames}/Camera7', f'Camera7_SyncFrame{n:05d}.png')
    frame8 = load_frame(f'{path_frames}/Camera8', f'Camera8_SyncFrame{n:05d}.png')
    # Assemble frames into one list
    frames = [frame2, frame3, frame4, frame6, frame7, frame8]
    # Return the combined frame
    return combine_frames(frames)


def plot_tableau(combined_frame):
    # figsize=[19.2, 10.8]
    fig, ax = plt.subplots(figsize=[3*10.8,2*19.2], dpi=100, frameon=False)
    ax.imshow(combined_frame)
    ax.axis('off')
    return fig


def fig2img(fig):
    """Convert a matlab figure to a numpy array of pixels (RGB)"""
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    return img[:, :, 0:3]


def combine_figs(figs):
    """Combine an array of 6 figures into one giant frame for visualization"""
    # Convert each figure into an image
    frames = list()
    for fig in figs:
        # Turn off frame
        # fig.set_frameon('False')
        # Make axes into the whole figure
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # Turn off axes
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # Convert the plot into an image
        frame = fig2img(fig) / 255.0
        frames.append(frame)
    # Combine the frames
    return combine_frames(frames)
