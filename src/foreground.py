"""
Michael S. Emanuel
Tue Dec 18 02:36:08 2018
"""

import os
import re
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from IPython.display import display
from image_background import load_frame
from am205_utils import arange_inc
from typing import List

# *************************************************************************************************
# Path for frames
path_frames = r'../sync_frames'
# Path for background frames
path_frames_bg = r'../frames/Background'
# Path for foreround frames
path_frames_fg = r'../foreground'

# Pixel count in the image
pixel_w: int = 1920
pixel_h: int = 1080
# Half size of pixels used frequently
pixel_hw: int = pixel_w // 2
pixel_hh: int = pixel_h // 2

# Default size for figures to match frames
figsize=[16.0, 9.0]


def plot_frame(frame):
    """Plot a frame in preparation to annotate it"""
    # Create axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xticks(arange_inc(0, 1920, 120))
    ax.set_yticks(arange_inc(0, 1080, 120))
    ax.set_xticklabels(arange_inc(0, 192, 12))
    ax.set_yticklabels(arange_inc(0, 108, 12))   
    # Display the image
    ax.imshow(frame)
    # Return the figure and axes
    return fig, ax


def frame_names(path_frames: str, camera_name: str) -> List[str]:
    """Return a list of file names for frames in this directory"""
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'
    # Files in this path; frames are named e.g. 'Camera1_Frame01234.png'
    fnames: List[str] = os.listdir(path)
    # Filter the list of frames to only those matching the pattern
    pattern: re.Pattern = re.compile(f'^{camera_name}_SyncFrame'+'(\d{5}).png$')
    # Return list of all file names matching this pattern
    return [fname for fname in fnames if pattern.match(fname) is not None]


camera_name = 'Camera3'

def save_fg(camera_name: str):
    frame_bg = load_frame(f'{path_frames_bg}', 'Camera3_median.png')
    
    # Find all the frames in this directory
    fnames: List[str] = frame_names(path_frames, camera_name)
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'
    # Path for foreground frames
    path_fg: str = f'{path_frames_fg}/{camera_name}'
    
    # Iterate over the frames for this camera
    for i, fname in enumerate(fnames[0:10]):
        # Load this frame
        frame = load_frame(path, fname)
        # Compute the foreground of the frame
        frame_fg = frame - frame_bg
        # Name of the foreground frame
        fname_fg = f'{path_fg}/{camera_name}_Foreground{i:05d}.png'
        # Save the foreground
        io.imsave(fname_fg, frame_fg)
    
    # Alternate version with white background
#        xxx = np.sum(frame_fg * frame_fg, axis=2)
#        mask = xxx < 0.02
#        frame_fgw = frame_fg.copy()
#        frame_fgw[mask] = 1.0
    

#    fig = plot_frame(frame - frame_bg)
#    display(fig)
#    plt.close(fig)
