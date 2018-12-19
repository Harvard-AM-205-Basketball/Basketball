"""
Harvard IACS Applied Math 205
Project: Basketball

track_ball.py: Track the position of the ball in the world frame (x, y, v)
based on its pixel positions from multiple cameras

Michael S. Emanuel
Tue Dec 18 23:48:03 2018
"""

import numpy as np
import pandas as pd
from camera_calibration import transforms
from am205_utils import range_inc
from typing import List

# *************************************************************************************************
# Path to ball_pixel
path_ball_pix: str = '../ball_pixel/'
# Camera numbers
camera_nums: np.ndarray = np.array([i for i in range_inc(8) if i != 5], dtype=np.int8)
# Camera names
camera_names: List[str] = [f'Camera{i}' for i in camera_nums]
# Number of cameras
camera_count: int = len(camera_nums)

# Frame rate
frame_rate: int = 30
# Number of frames
frame_count: int = 4391

# Array of time vs fn
tt = np.array(np.arange(frame_count)) / frame_rate


# *************************************************************************************************
def load_data():
    # Load CSV file for each camera
    ball_pix_tbl = dict()
    for camera_name in camera_names:
        # Read the ball pixel positions from the CSV file
        ball_pix_j = pd.read_csv(f'{path_ball_pix}/{camera_name}.csv', names = ['t', 'u', 'v'])
        # Add a column with the frame numbers
        ball_pix_j['n'] = np.round(frame_rate*ball_pix_j.t).astype(np.int32)
        ball_pix_tbl[camera_name] = ball_pix_j
        
    # Add integer keys as quicker aliases
    for i in camera_nums:
        ball_pix_tbl[i] = ball_pix_tbl[f'Camera{i}']
    
    
    # Assemble all available positions into a numpy array; use NaN where position unavailable
    ball_pix_mat: np.ndarray = np.full([frame_count, camera_count, 2], np.nan)
    # Which cameras are available at each frame?
    mask_mat: np.ndarray = np.zeros([frame_count, camera_count], dtype=bool)
    
    # Fill in one column for each camera
    for j, camera_name in enumerate(camera_names):
        # Look up this ball pixel dataframe, ball_pix_j 
        ball_pix_j = ball_pix_tbl[camera_name]
        # Get the frame numbers, u, and v for this camera
        n = ball_pix_j.n
        u = ball_pix_j.u
        v = ball_pix_j.v
        # Fill in the u and v coordinates on the third axis of ball_pix
        ball_pix_mat[n, j, 0] = u
        ball_pix_mat[n, j, 1] = v
        # Fill in ball_mask
        mask_mat[n, j] = (~np.isnan(u)) & (~np.isnan(v))

    # Return the matrix of ball pixel positions and the mask of camera availability
    return ball_pix_mat, mask_mat
    

# *************************************************************************************************
# Load the data
ball_pix_mat, mask_mat = load_data()

# start of shot
n = 2394
t = n / frame_rate
ball_pix = ball_pix_mat[n]
mask = mask_mat[n]
cams = camera_nums[mask]

