"""
Michael S. Emanuel
Tue Dec 18 02:36:08 2018
"""

import os
import re
import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
# from IPython.display import display
from image_background import load_frame_i
from joblib import Parallel, delayed
from tqdm import tqdm
from am205_utils import range_inc, arange_inc
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

# A white frame
white_frame = np.ones((1080,1920,3), dtype=np.uint8) * 255


# *************************************************************************************************
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


def calc_fg(frame, fgmask):
    """Apply a mask to a frame to get the foreground"""
    return cv2.bitwise_and(frame, frame, mask=fgmask)


def calc_fgw(frame, fgmask):
    """Apply a mask to a frame to get the foreground against a white background"""
    return cv2.bitwise_or(frame, white_frame, mask=255-fgmask)

def save_fg(camera_name: str):
    """Save the foregrounds of these frames"""
    # Find all the frames in this directory
    fnames: List[str] = frame_names(path_frames, camera_name)
    # Path with frames for this camera
    path: str = f'{path_frames}/{camera_name}'
    # Path for foreground frames
    path_fg: str = f'{path_frames_fg}/{camera_name}'
    
    # Initialize an opencv background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    # Iterate over the frames for this camera
    for i, fname in enumerate(tqdm(fnames)):
        # Load this frame
        frame_i = load_frame_i(path, fname)
        # frame = (frame_i / 255.0).astype(np.float32)
        # Apply the background subtractor
        fgmask = fgbg.apply(frame_i)
        # Compute the foreground of the frame
        frame_fg_i = calc_fg(frame_i, fgmask)
        # Name of the foreground frame
        fname_fg = f'{path_fg}/{camera_name}_Foreground{i:05d}.png'
        # Save the foreground
        io.imsave(fname_fg, frame_fg_i)


# *************************************************************************************************
def main():
    # List of Camera names
    camera_names: List[str] = [f'Camera{n}' for n in range_inc(1, 8) if n != 5]
    # Number of cameras
    camera_count: int = len(camera_names)

    # Iterate over all the cameras
    # for camera_name in camera_names:
    #    print(f'Extracting foreground for {camera_name}...')
    #     save_fg(camera_name)
    Parallel(n_jobs=camera_count, prefer='threads')(
        delayed(save_fg)(camera_name)
        for camera_name in camera_names)


if __name__ == '__main__':
    main()
