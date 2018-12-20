"""
Harvard IACS Applied Math 205
Project: Basketball

track_ball.py: Track the position of the ball in the world frame (x, y, v)
based on its pixel positions from multiple cameras

Michael S. Emanuel
Tue Dec 18 23:48:03 2018
"""

import sys
import os
from joblib import Parallel, delayed
import numpy as np
from numpy import pi
import pandas as pd
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from textwrap import dedent
from tqdm import tqdm
import warnings
from camera_transform import focus, pixel_size
from camera_calibration import transforms as transforms_cal, cam_pos_tbl, zoom_tbl
from image_utils import load_frame, combine_figs
from am205_utils import range_inc, arange_inc
from typing import List

# *************************************************************************************************
# Global variables
# Path to ball_pixel
path_ball_pix: str = '../ball_pixel/'
# Path with synchronized frames
path_sync_frames = r'../sync_frames'
# Path with ball frames
path_ball_frames = r'../ball_frames'
# Path for ball tableau
path_ball_tableau = r'../ball_tableau'

# Camera numbers
camera_nums: np.ndarray = np.array([i for i in range_inc(8) if i != 5], dtype=np.int8)
# Camera names
camera_names: List[str] = [f'Camera{i}' for i in camera_nums]
# Number of cameras
camera_count: int = len(camera_nums)
# Mask to specify all cameras
mask_all = np.ones(camera_count, dtype=bool)

# Frame rate
frame_rate: int = 30
# Number of frames
frame_count: int = 4391
# Array of time vs fn
tt = np.array(np.arange(frame_count)) / frame_rate

# Pixel count in the image
pixel_w: int = 1920
pixel_h: int = 1080
# Half size of pixels used frequently
pixel_hw: int = pixel_w // 2
pixel_hh: int = pixel_h // 2

# Weights on the cameras
weight = np.ones(camera_count)

# Get the pixel transforms for all the cameras
# transforms[i] is transform_fg for that camera; maps position to floating point pixel (u, v)
transforms = np.empty(7, dtype=object)
# transforms_xy = np.empty(7, dtype=object)
for i, camera_num in enumerate(camera_nums):
    transforms[i] = transforms_cal[camera_num][2]

# Build 7x3 matrix of camera positions and array of 7 zooms
cam_pos_mat = np.zeros((7,3))
zooms = np.zeros(7)
for i, camera_name in enumerate(camera_names):
    cam_pos_mat[i, :] = cam_pos_tbl[camera_name]
    zooms[i] = zoom_tbl[camera_name]

# Default size for figures to match frames using 100 dpi
figsize=[19.2, 10.8]
dpi=100

# Radius of a basketball: NBA standard is 29.5 inches circumference.  Need this in feet.
R: float = 29.5 / (2*pi) / 12.0

# *************************************************************************************************
# Set plot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'figure.max_open_warning':256})
# Ignore warnings
warnings.filterwarnings("ignore")

# *************************************************************************************************
def load_data():
    # Load CSV file for each camera
    ball_pix_tbl = dict()
    for camera_name in camera_names:
        # Read the ball pixel positions from the CSV file
        ball_pix_df = pd.read_csv(f'{path_ball_pix}/{camera_name}.csv', names = ['t', 'x', 'y'])
        # Add a column with the frame numbers
        ball_pix_df['n'] = np.round(frame_rate*ball_pix_df.t).astype(np.int32)
        # Add columns to compute (u, v) from x and y
        ball_pix_df['u'] = +ball_pix_df.x + 320.0
        ball_pix_df['v'] = -ball_pix_df.y + 240.0        
        # Save this frame to the ball_pix_tbl dictionary
        ball_pix_tbl[camera_name] = ball_pix_df
        
    # Add integer keys as quicker aliases
    for i in camera_nums:
        ball_pix_tbl[i] = ball_pix_tbl[f'Camera{i}']
        
    # Assemble all available positions into a numpy array; use NaN where position unavailable
    ball_pix_mat: np.ndarray = np.full([frame_count, camera_count, 2], np.nan)
    # Which cameras are available at each frame?
    mask_mat: np.ndarray = np.zeros([frame_count, camera_count], dtype=bool)
    
    # Fill in one column for each camera
    for j, camera_name in enumerate(camera_names):
        # Look up this ball pixel dataframe, ball_pix
        ball_pix_df = ball_pix_tbl[camera_name]
        # Get the frame numbers, u, and v for this camera
        n = ball_pix_df.n
        u = ball_pix_df.u
        v = ball_pix_df.v
        # Fill in the u and v coordinates on the third axis of ball_pix
        ball_pix_mat[n, j, 0] = u
        ball_pix_mat[n, j, 1] = v
        # Fill in ball_mask
        mask_mat[n, j] = (~np.isnan(u)) & (~np.isnan(v))

    # Return the matrix of ball pixel positions and the mask of camera availability
    return ball_pix_mat, mask_mat


# *************************************************************************************************
# Assemble all of the transorms into one big function from world coordinates to 7 pixels
def world2pix(pos: np.ndarray, mask: np.ndarray):
    """Map a 3D world location to (u,v) coordinates for all 7 cameras"""
    # Count the number of cameras included in this mask
    n: int = np.sum(mask)
    # Initialize Nx2 matrix of zeros
    pix_mat = np.zeros((n,2))
    # Fill in the rows
    for i, transform in enumerate(transforms[mask]):
        pix_mat[i,:] = transform(pos)
    return pix_mat


def make_difference_func(ball_pix: np.ndarray, mask: np.ndarray):
    """Make an function computing the difference between calculated and observed pixels"""

    def difference_func(pos: np.ndarray):
        """The objective function that is optimized w.r.t. pos"""
        # Apply world2pix to produce pixel locations
        calc_pix = world2pix(pos, mask)
        # Return the difference with the observed pixels
        return calc_pix - ball_pix

    # Return the assembled difference function
    return difference_func


def make_objective_func(ball_pix: np.ndarray, mask: np.ndarray, weight: np.ndarray):
    """Make an optimization objective function"""
    
    # The objective function is the sum of squares difference with these weights
    def objective_func(pos):
        # Generate the pixel difference function
        difference_func = make_difference_func(ball_pix, mask)
        # Compute the difference        
        diff = difference_func(pos)
        # Compute the weighted sum of squares
        diff_by_cam = np.sum(diff * diff, axis=1)
        return np.sum(diff_by_cam * weight[mask])

    # Return the assembled optimization objective function
    return objective_func


# *************************************************************************************************
def plot_frame(frame: np.ndarray):
    """Plot a frame in preparation to annotate it"""
    # Create a figure with NO FRAME
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    # Create an axis that is the WHOLE FIGURE
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # Turn off display of axes (ticks etc)
    ax.set_axis_off()
    # Now add this axis to the figure
    fig.add_axes(ax)
    
    # Proceed as normal, setting limits etc. for plotting
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xticks(arange_inc(0, 1920, 120))
    ax.set_yticks(arange_inc(0, 1080, 120))
    ax.set_xticklabels(arange_inc(0, 192, 12))
    ax.set_yticklabels(arange_inc(0, 108, 12))
    
    # Display the frame as an image
    ax.imshow(frame)
    # Return the figure and axes
    return fig, ax


def plot_tableau(frame: np.ndarray):
    """Plot a frame in preparation to annotate it"""
    # Create a figure with NO FRAME
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    fig = plt.figure(figsize=[19.2*2, 10.8*3], dpi=dpi, frameon=False)
    # Create an axis that is the WHOLE FIGURE
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    # Turn off display of axes (ticks etc)
    ax.set_axis_off()
    # Now add this axis to the figure
    fig.add_axes(ax)  
    # Display the frame as an image
    ax.imshow(frame)
    # Return the figure and axes
    return fig, ax


def frame_overlay(frames: List[np.ndarray], calc_pix_all: np.ndarray, ball_pos: np.ndarray, j: int):
    """Overlay frame j with the ball"""
    # The selected frame
    frame = frames[j]
    # Plot the frame
    fig, ax = plot_frame(frame)
    # Extract ball u, v for this frame from calc_pix_all
    u, v = calc_pix_all[j]
    # Position of this camera
    cam_pos = cam_pos_mat[j, :]
    # Effective focus for this camera
    focus_eff = focus * zooms[j]
    # Distance from camera to ball position
    ball_dist = np.linalg.norm(ball_pos - cam_pos)
    # Radius of this circle in PIXELS 
    ball_radius = R * (focus_eff / ball_dist) / pixel_size
    # Plot the circle on the frame image
    circle = Circle((u,v), ball_radius, fill=False, color='r', linewidth=2.0)
    ax.add_patch(circle)
    # Return the modified figure and axis
    return fig, ax


def frames_overlay(n: int, ball_pos, mask):
    """Overlay frame n with the projected ball location"""
    # Load the frames
    frame1 = load_frame(f'{path_sync_frames}/Camera1', f'Camera1_SyncFrame{n:05d}.png')
    frame2 = load_frame(f'{path_sync_frames}/Camera2', f'Camera2_SyncFrame{n:05d}.png')
    frame3 = load_frame(f'{path_sync_frames}/Camera3', f'Camera3_SyncFrame{n:05d}.png')
    frame4 = load_frame(f'{path_sync_frames}/Camera4', f'Camera4_SyncFrame{n:05d}.png')
    frame6 = load_frame(f'{path_sync_frames}/Camera6', f'Camera6_SyncFrame{n:05d}.png')
    frame7 = load_frame(f'{path_sync_frames}/Camera7', f'Camera7_SyncFrame{n:05d}.png')
    frame8 = load_frame(f'{path_sync_frames}/Camera8', f'Camera8_SyncFrame{n:05d}.png')
    
    # Load frames for all seven cameras
    frames = [frame1, frame2, frame3, frame4, frame6, frame7, frame8]

    # Compute implied pixel positions of ball 
    calc_pix_all = world2pix(ball_pos, mask_all)
    
    # Generate a list of frames with the ball
    figs = 7 * [None]
    for j, plot_j in enumerate(mask):
        # If this frame was included in the calibration, overlay the ball position
        if plot_j:
            fig, ax = frame_overlay(frames, calc_pix_all, ball_pos, j)
        # Otherwise just plot the frame without any annotation
        else:
            fig, ax = plot_frame(frames[j])
        # Turn off the axis
        ax.axis('off')
        # Put this frame in the list
        figs[j] = fig
    return figs


def save_ball_figs(figs, n: int):
    """Save ball frames amd tableau"""
    # The camera number and names corresponding to these figures
    camera_nums = [1, 2, 3, 4, 6, 7, 8]
    camera_names = [f'Camera{cn}' for cn in camera_nums]
    # Enumerate through each figure in the collection and save it to the ball_frames directory
    for i, fig in enumerate(figs):
        # Camera name corresponding to this ball figure
        camera_name = camera_names[i]
        # The name of this file
        fname = f'{path_ball_frames}/{camera_name}/{camera_name}_BallFrame{n:05d}.png'
        fig.savefig(fname, dpi=dpi)
    # Combine 6 frames into one tableau; skip camera 1
    tableau_frame = combine_figs(figs[1:7])
    tableau_fig, tableau_ax = plot_tableau(tableau_frame)
    # Save the tableau figure
    fname_tab = f'{path_ball_tableau}/BallTableau{n:05d}.png'
    tableau_fig.savefig(fname_tab, dpi=dpi)
    # Close all the ball figures
    for fig in figs:
        plt.close(fig)
    # Close the tableau fig
    plt.close(tableau_fig)


# *************************************************************************************************
def track_frame(n: int):
    """Track the ball in one frame"""
    # Get the mask
    mask = mask_mat[n]
    # Don't use camera 1 - not a good angle
    mask[0] = False
    # Get the ball pixels
    ball_pix = ball_pix_mat[n][mask]
    
    # Use a neutral initial guess - top of the key for a half-court game
    # elevation 5.0 feet is half way between the floor and the rim
    pos0 = np.array([0.0, 26.0, 5.0])
    
    # Create optimization objective function for this frame
    objective_func = make_objective_func(ball_pix, mask, weight)
    
    # Run the optimization
    res = minimize(fun=objective_func, x0=pos0)
    ball_pos = res['x']
    
    # Compute time
    t = n / frame_rate
    
    # Extract x, y, z
    x, y, z = ball_pos

    # Integer key for mask
    mask2int = np.array([2**i for i in range(7)])
    mask_int = np.sum(mask * mask2int)

    # Save this row to the dataframe
    ball_pos_df = pd.DataFrame(columns=['n', 't', 'x', 'y', 'z', 'mask'])
    ball_pos_df = ball_pos_df.append({'n':n, 't':t, 'x':x, 'y':y, 'z':z, 'mask': mask_int}, ignore_index=True)    
    # Save the dataframe
    fname_df = f'../calculations/ball_pos_{n:05d}.csv'
    ball_pos_df.to_csv(fname_df)

    # Generate figures (ball for each camera plus one tableau)
    ball_figs = frames_overlay(n, ball_pos, mask)
    save_ball_figs(ball_figs, n)


def report(n: int):
    """Compute the implied pixel locations and difference with detailed reporting"""
    # Get the mask
    mask = mask_mat[n]
    mask[0] = False

    # Get the ball pixels
    ball_pix = ball_pix_mat[n][mask]
    cams = camera_nums[mask]
    
    # Use a neutral initial guess - top of the key for a half-court game
    # elevation 5.0 feet is half way between the floor and the rim
    pos0 = np.array([0.0, 26.0, 5.0])
    
    # Solve for ball position
    difference_func = make_difference_func(ball_pix, mask)
    # Create functions for difference in pixels and objective
    objective_func = make_objective_func(ball_pix, mask)
    
    # Run the optimization
    res = minimize(fun=objective_func, x0=pos0)
    ball_pos = res['x']
    
    # Compute time
    t = n / frame_rate
    
    # Extract x, y, z
    x, y, z = ball_pos
    
    # Report the results
    print(f'Solved for ball position at frame {n} / t={t:0.3f}.')
    # calc_pix = world2pix(ball_pos, mask)
    pixel_diff = np.round(difference_func(ball_pos))
    print('Computed Ball Position:')
    print(ball_pos)
    print(f'Cameras: {cams}')
    print('Pixel Difference for cameras used:')
    print(pixel_diff)


def track_frames(frame_nums, progress_bar: bool = False):
    """Process a batch of frames"""
    # Wrap the frame_nums in tqdm if progress_bar was specified
    frame_iter = tqdm(frame_nums) if progress_bar else frame_nums    
    for n in frame_iter:
        # The filename
        fname = f'{path_ball_tableau }/BallTableau{n:05d}.png'
        # If this file already exists, skip it and continue
        if os.path.isfile(fname):
            continue
        track_frame(n)

    
# *************************************************************************************************
# Load the data into the global name space
ball_pix_mat, mask_mat = load_data()

def main():
    # Load the data
    ball_pix_mat, mask_mat = load_data()
    
    # Range of frames to process
    n0: int
    n1: int
    # Number of parallel threads to use
    jobs: int
    
    # Process command line arguments
    argv: List[str] = sys.argv
    argc: int = len(sys.argv)-1
    usage_str = dedent(
    """
    python ball_track.py 
        process all frames        
    python ball_track.py n0 n1
        track frames in [n0, n1)
    python ball_track.py n0 n1 j
        track frames in [n0, n1) using jobs threads
    """)
    try:
        if argc == 0:
            n0 = 0
            n1 = frame_count
            jobs = 1
        elif argc == 1:
            n0 = 0
            n1 = frame_count
            jobs = int(argv[1])
        elif argc == 2:
            n0 = int(argv[1])
            n1 = int(argv[2])
            jobs = 1
        elif argc == 3:
            n0 = int(argv[1])
            n1 = int(argv[2])
            jobs = int(argv[3])
        else:
            raise RuntimeError
    except:
        print(f'Error in arguments for make_tableaux.py.  argc={argc}, argv={argv}.')
        print(usage_str)
        exit()
    print(f'Processing frames from {n0} to {n1} on {jobs} threads.')
    
    # Generate list of candidate frame numbers; only those where the file
    # does not already exist (don't want to process duplicates)
    # frame_nums = list(range(n0, n1))
    frame_nums = list()
    for n in range(n0, n1):
        # The filename
        fname = f'{path_ball_tableau }/BallTableau{n:05d}.png'
        # If this file already exists, skip it and continue
        if not os.path.isfile(fname):
            frame_nums.append(n)
    
    # Report number of tracks
    num_left = len(frame_nums)
    num_skipped = (n1 - n0) - num_left
    print(f'Identified {num_left} frames to process; skipping {num_skipped}.')
            
    # Split up the frames for apportionment to different threads
    job_tbl = dict()
    for k in range(jobs):
        job_tbl[k] = [n for n in frame_nums if n % jobs == k]
        
    # List of arguments for parallel job
    args = [(job_tbl[jn], jn == 1) for jn in range(jobs)]
    
    # Run these jobs in parallel if jobs > 1
    if jobs > 1:
        Parallel(n_jobs=jobs, prefer='threads')(
                delayed(track_frames)(frame_nums, progress_bar)
                for frame_nums, progress_bar in args)
    # Otherwise run this single threaded
    else:
        track_frames(frame_nums, True)

if __name__ == '__main__':
    main()
