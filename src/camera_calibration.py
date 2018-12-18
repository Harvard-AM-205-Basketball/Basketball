"""
camera_transform.py


Michael S. Emanuel
Mon Dec 17 21:38:55 2018
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from image_background import load_frame
from basketball_court import make_court_lines
from camera_transform import make_transform_uv
from am205_utils import arange_inc


# *************************************************************************************************
# Set plot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})

# Default size for figures to match frames
figsize=[16.0, 9.0]

# Path for background frames
path_frames = r'..\frames\Background'

# Pixel count in the image
pixel_w: int = 1920
pixel_h: int = 1080
# Half size of pixels used frequently
pixel_hw: int = pixel_w // 2
pixel_hh: int = pixel_h // 2


# *************************************************************************************************
def plot_frame(frame_name: str):
    """Plot a frame in preparation to annotate it"""
    frame = load_frame(path_frames, frame_name)
    # Create axes
    fig, ax = plt.subplots(figsize=figsize)
    # y_ticks = np.array([0, 120, 240, 360, 480, 540, 600, 720, 820, 960, 1080])
    # y_ticklabels = y_ticks // 10
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xticks(arange_inc(0, 1920, 120))
    ax.set_yticks(arange_inc(0, 1080, 120))
    # ax.set_yticks(y_ticks)
    ax.set_xticklabels(arange_inc(0, 192, 12))
    ax.set_yticklabels(arange_inc(0, 108, 12))
    # ax.set_yticklabels(y_ticklabels)
    # ax.grid()
    
    # Add vertical and horizontal lines for the center
    ax.axhline(pixel_hh, color='gray', linewidth=1.0)
    ax.axvline(pixel_hw, color='gray', linewidth=1.0)
    # Add a "cross hair" in the middle
    ax.plot(pixel_hw, pixel_hh, marker='+', markersize=20, markeredgewidth=2.0, color='r')
    # Display the image
    ax.imshow(frame)
    # Return the figure and axes
    return fig, ax


def annotate_frame_line(ax, pixels: np.ndarray, color: str):
    """Annotate a frame with a batch of pixels"""
    # Plot the pixels on the axis
    ax.plot(pixels[:,0], pixels[:,1], color=color, linewidth=1.0)


def annotate_frame_dots(ax, pixels: np.ndarray, color: str):
    """Annotate a frame with a batch of pixels"""
    # Get u and v from pixels so we can filter them to [0, 1920) x [0, 1080)
    u = pixels[:,0]
    v = pixels[:,1]
    # The filter
    mask = (0 <= u) & (u < pixel_w) & (0 <= v) & (v < pixel_h)
    # Plot the pixels on the axis
    ax.plot(pixels[mask,0], pixels[mask,1], color=color, linewidth=0, marker='o', markersize=1)



# Visual features for calibration
court_lines = make_court_lines()
perimeter = court_lines['perimeter']
key_box_N = court_lines['key_box_N']
key_circle_N = court_lines['key_circle_N']
three_point_N = court_lines['three_point_N']
vertical_N = court_lines['vertical_N']
backboard_N = court_lines['backboard_N']
backboard_in_N = court_lines['backboard_in_N']
rim_N = court_lines['rim_N']


# *************************************************************************************************
def calibrate_cam2():
    """Calibration for camera 2"""
    # Name of this camera
    camera_name = 'Camera2'
    
    # Position of camera (adjusted for height of tripod)
    cam_pos = np.array([21.0, 21.0, 5.2])
    # Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
    cam_point = np.array([-2.0, 41.0, 6.5])
    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.00)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    rim_uv = transform(rim_N)
    
    # Make the calibration plot
    fig, ax = plot_frame(f'{camera_name}_median.png')
    # Overlay the pixels
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    annotate_frame_line(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig('../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


def calibrate_cam3():
    """Calibration for camera 3"""
    # Name of this camera
    camera_name = 'Camera3'
    
    # Position of camera (adjusted for height of tripod)
    # cam_pos = np.array([22.0, -1.5, 5.2])
    cam_pos = np.array([22.0, 0.0, 5.2])
    # Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
    cam_point = np.array([0.0, 41.0, 8.0])
    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.00)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    key_box_uv = transform(key_box_N)
    key_circle_uv = transform(key_circle_N)
    three_point_uv = transform(three_point_N)
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    
    # Make the calibration plot
    fig, ax = plot_frame(f'{camera_name}_median.png')
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_line(ax, key_box_uv, 'r')
    annotate_frame_line(ax, key_circle_uv, 'r')
    annotate_frame_line(ax, three_point_uv, 'r')
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


def calibrate_cam7():
    """Calibration for camera 7"""
    # Name of this camera
    camera_name = 'Camera7'
    
    # Position of camera (adjusted for height of tripod)
    cam_pos = np.array([-21.0, 0.0, 5.2])
    # Camera 7 is pointed at
    cam_point = np.array([8.3, 41.0, 5.0])
    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.05)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    key_box_uv = transform(key_box_N)
    key_circle_uv = transform(key_circle_N)
    three_point_uv = transform(three_point_N)
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    
    # Make the calibration plot
    fig, ax = plot_frame(f'{camera_name}_median.png')
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_line(ax, key_box_uv, 'r')
    annotate_frame_line(ax, key_circle_uv, 'r')
    annotate_frame_line(ax, three_point_uv, 'r')
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


# *************************************************************************************************
# Run the calibration
# calibrate_cam2()
calibrate_cam3()
# calibrate_cam7()
