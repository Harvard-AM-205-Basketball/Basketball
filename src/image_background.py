"""
Harvard IACS Applied Math 205
Project: Basketball

image_background.py
Extract the background from a sequence of frames taken by the same camera.
Background is inferred as either the mean or median value for each pixel.

Usage:
python image_background.py 
    process all cameras, all frames
python image_background.py 10
    process all cameras, first 10 frames for each camera only
python image_backgroud.py c0 c1
    calculate background for cameras numbered [c0, ..., c1]; all frames
python image_backgroud.py c0 c1 mf
    calculate background for cameras numbered [c0, ..., c1]; max_frames

Michael S. Emanuel
Sat Dec 15 08:48:28 2018
"""

import sys
import numpy as np
from skimage import io
from IPython.display import display
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
from image_utils import load_frames
from am205_utils import range_inc
from typing import List, Optional

# *************************************************************************************************
# silence irrelevant user warnings
warnings.filterwarnings("ignore", category=UserWarning)


# *************************************************************************************************
def calc_mean_frame(frames) -> np.ndarray:
    """Compute the mean frame from an array of frames in 8 bit integer format"""
    # For each pixel, compute the mean and median
    return np.mean(frames, axis=0, dtype=np.float32) / 255.0


def calc_median_frame(frames) -> np.ndarray:
    """Compute the median frame from an array of frames in 8 bit integer format"""
    # For each pixel, compute the mean and median
    ans = np.zeros(shape=frames.shape[1:4], dtype=np.float32)
    # Compute the median with the answer saved in-place to out; allows specification of float32 type
    np.median(frames, axis=0, out=ans)
    return  ans / 255.0


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
    # Range of cameras to process
    c0: int
    c1: int
    # Set maximum frames per camera
    max_frames: Optional[int]

    # Process command line arguments
    argv: List[str] = sys.argv
    argc: int = len(sys.argv)-1
    usage_str = \
"""
python image_background.py 
    process all cameras, all frames
python image_background.py 10
    process all cameras, first 10 frames for each camera only
python image_backgroud.py c0 c1
    calculate background for cameras numbered [c0, ..., c1]; all frames
python image_backgroud.py c0 c1 mf
    calculate background for cameras numbered [c0, ..., c1]; max_frames
"""
    try:
        if argc == 0:
            c0 = 1
            c1 = 8
            max_frames = None
        elif argc == 1:
            c0 = 1
            c1 = 8
            max_frames = int(argv[1])
        elif argc == 2:
            c0 = int(argv[1])
            c1 = int(argv[2])
            max_frames = None
        elif argc == 3:
            c0 = int(argv[1])
            c1 = int(argv[2])
            max_frames = int(argv[3])
        else:
            raise RuntimeError
    except:
        print(f'Error in arguments for image_background.py.  argc={argc}, argv={argv}.')
        print(usage_str)
        exit()
    print(f'Processing cameras from {c0} to {c1}; max_frames={max_frames}.')


    # Path to frames directory
    path_frames: str = r'../frames'
    # Path to directory of background frames
    path_background: str = r'../frames/Background'
    # List of Camera names
    camera_names: List[str] = [f'Camera{n}' for n in range_inc(c0, c1) if n != 5]
    # Number of cameras
    camera_count: int = len(camera_names)
    
    # Figure size for displaying mean and median
    figsize=[19.2, 10.8]
    
    # Compute mean and median frame in parallel
    # https://joblib.readthedocs.io/en/latest/parallel.html
    print(f'Running {camera_count} parallel jobs on threads (1 for each camera)...')
    background = Parallel(n_jobs=camera_count, prefer='threads')(
        delayed(process_one_camera)(path_frames, camera_name, max_frames)
        for camera_name in camera_names)
    
    # Save and display the background frames (mean and median)
    for i, camera_name in enumerate(camera_names):
        # Look up the results for this camera from parallel job
        mean_frame, median_frame = background[i]
    
        # Display and save the mean frame
        print(f'Mean Frame for {camera_name}')
        fig = plt.figure(figsize=figsize)
        io.imshow(mean_frame)
        display(fig)
        plt.close(fig)
        # Save the mean frame
        # io.imsave(f'{path}/{camera_name}_mean.png', mean_frame)
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
        # io.imsave(f'{path}/{camera_name}_median.png', median_frame)
        io.imsave(f'{path_background}/{camera_name}_median.png', median_frame)
        np.save(f'{path_background}/{camera_name}_median.npy', median_frame)

if __name__ == '__main__':
    main()
