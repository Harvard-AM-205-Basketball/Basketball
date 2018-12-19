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
def plot_frame(frame: np.ndarray):
    """Plot a frame in preparation to annotate it"""
    # frame = load_frame(path_frames, frame_name)
    # Create axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_xticks(arange_inc(0, 1920, 120))
    ax.set_yticks(arange_inc(0, 1080, 120))
    ax.set_xticklabels(arange_inc(0, 192, 12))
    ax.set_yticklabels(arange_inc(0, 108, 12))
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
center_circle = court_lines['center_circle']
key_box_N = court_lines['key_box_N']
key_circle_N = court_lines['key_circle_N']
three_point_N = court_lines['three_point_N']
vertical_N = court_lines['vertical_N']
backboard_N = court_lines['backboard_N']
backboard_in_N = court_lines['backboard_in_N']
rim_N = court_lines['rim_N']

key_box_S = court_lines['key_box_S']
key_circle_S = court_lines['key_circle_S']
three_point_S = court_lines['three_point_S']
vertical_S = court_lines['vertical_S']
backboard_S = court_lines['backboard_S']
backboard_in_S = court_lines['backboard_in_S']
rim_S = court_lines['rim_S']


# *************************************************************************************************
def calibrate_cam3():
    """Calibration for camera 3"""
    # Name of this camera
    camera_name = 'Camera3'
    
    # Position of camera (adjusted for height of tripod)
    cam_pos = np.array([20.0, 0.0, 5.2])
    # Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
    cam_point = np.array([0.0, 38.3, 8.0])
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
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
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
    cam_pos = np.array([-20.0, 0.0, 5.2])
    # Camera 7 is pointed at
    cam_point = np.array([9.1, 38.0, 4.2])

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
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
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
def calibrate_cam2():
    """Calibration for camera 2"""
    # Name of this camera
    camera_name = 'Camera2'
    
    # Position and orientation of camera
    # cam_pos = np.array([20.0, 43.5, 5.5])
    # cam_point = np.array([0.0, 38.0, 8.1])
    cam_pos = np.array([20.0, 43.0, 5.5])
    cam_point = np.array([-20, 33.2, 10.5])

    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.00)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    rim_uv = transform(rim_N)
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
    # Overlay the pixels
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    annotate_frame_line(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)



def calibrate_cam8():
    """Calibration for camera 8"""
    # Name of this camera
    camera_name = 'Camera8'
    
    # Position and orientation of camera
    cam_pos = np.array([-20.0, 43.0, 5.5])
    # cam_point = np.array([1.0, 34.5, 7.5])
    cam_point = np.array([20.0, 23.0, 10.0])

    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.00)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    three_point_N_uv = transform(three_point_N)
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    rim_uv = transform(rim_N)
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_line(ax, three_point_N_uv, 'r')
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    annotate_frame_line(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)

# *************************************************************************************************
def calibrate_cam4():
    """Calibration for camera 4"""
    # Name of this camera
    camera_name = 'Camera4'
    
    # Position and orientation of camera
    cam_pos = np.array([20.0, -42.0, 5.7])
    # cam_point = np.array([-1.2, 49.0, 5.2])
    cam_point = np.array([-1.3, 49.0, cam_pos[2]-0.8])

    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 2.02)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    center_circle_uv = transform(center_circle)
    key_box_N_uv = transform(key_box_N) 
    key_circle_N_uv = transform(key_circle_N) 
    three_point_N_uv = transform(three_point_N)
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    rim_uv = transform(rim_N)
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_line(ax, center_circle_uv, 'r')
    annotate_frame_line(ax, key_box_N_uv, 'r')
    annotate_frame_line(ax, key_circle_N_uv, 'r')
    annotate_frame_line(ax, three_point_N_uv, 'r')    
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    annotate_frame_line(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


def calibrate_cam6():
    """Calibration for camera 6"""
    # Name of this camera
    camera_name = 'Camera6'
    
    # Position and orientation of camera
    cam_pos = np.array([-20.0, -42.0, 5.6])
    cam_point = np.array([2.6, 49.0, cam_pos[2]-0.3])

    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 2.22)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    center_circle_uv = transform(center_circle)
    key_box_N_uv = transform(key_box_N) 
    key_circle_N_uv = transform(key_circle_N) 
    three_point_N_uv = transform(three_point_N)
    vertical_uv = transform(vertical_N)
    backboard_uv = transform(backboard_N)
    backboard_in_uv = transform(backboard_in_N)   
    rim_uv = transform(rim_N)
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_line(ax, center_circle_uv, 'r')
    annotate_frame_line(ax, key_box_N_uv, 'r')
    annotate_frame_line(ax, key_circle_N_uv, 'r')
    annotate_frame_line(ax, three_point_N_uv, 'r')    
    annotate_frame_line(ax, vertical_uv, 'r')
    annotate_frame_line(ax, backboard_uv, 'r')
    annotate_frame_line(ax, backboard_in_uv, 'r')
    annotate_frame_line(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


# *************************************************************************************************
def calibrate_cam1():
    """Calibration for camera 1"""
    # Name of this camera
    camera_name = 'Camera1'
    
    # Position and orientation of camera
    cam_pos = np.array([0.0, 38.0 + 2.8, 10.0 +0.48])
    cam_point = np.array([1.0, -48.0, 8.0])

    # Transform for camera 3 (to pixel space)
    transform = make_transform_uv(cam_pos, cam_point, 1.00)
    
    # Apply the transform for camera 3 to various shapes of interest
    # Convert the xy positions to uv pixel locations
    perimeter_uv = transform(perimeter)
    center_circle_uv = transform(center_circle)
    key_box_S_uv = transform(key_box_S)
    key_circle_S_uv = transform(key_box_S)
    three_point_S_uv = transform(three_point_S)
    rim_uv = transform(rim_N)
    
    # Load the frame
    frame = load_frame(path_frames, f'{camera_name}_median.png')

    # Make the calibration plot
    fig, ax = plot_frame(frame)
    # Overlay the pixels
    annotate_frame_dots(ax, perimeter_uv, 'r')
    annotate_frame_dots(ax, center_circle_uv, 'r')
    annotate_frame_dots(ax, key_box_S_uv, 'r')
    annotate_frame_dots(ax, key_circle_S_uv, 'r')
    annotate_frame_dots(ax, three_point_S_uv, 'r')    
    annotate_frame_dots(ax, rim_uv, 'r')
    # Display and save the figure    
    display(fig)
    fig.savefig(f'../figs/{camera_name}_calibration.png', bbox_inches='tight')
    plt.close(fig)


# *************************************************************************************************
# Run the calibration
calibrate_cam3()
calibrate_cam7()
calibrate_cam2()
calibrate_cam8()
calibrate_cam4()
calibrate_cam6()
calibrate_cam1()
