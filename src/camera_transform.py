"""
Harvard IACS Applied Math 205
Project: Basketball

camera_transform.py
Transform images between three coordinate system:
    world  3D (U, V, W)
    camera 3D (X, Y, Z)
    camera image plane 2D (x, y)
    camera pixed index 2D (u, v)

Originally developed by Nicholas Beasley
Adapted by Michael S. Emanuel
Mon Dec 17 12:54:44 2018
"""

import numpy as np
# from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
from am205_utils import arange_inc
from basketball_court import make_court_lines
from image_background import load_frame

# *************************************************************************************************
# Set plot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})

# Default size for figures to match frames
figsize=[16.0, 9.0]

# Focal length (0.024147 ft is from manufacturer), changes with zoom
focus: float = 0.024147

# Pixel size
pixel_size: float = 1.02E-5
pixel_size_x: float = pixel_size
pixel_size_y: float = pixel_size

# Pixel count in the image
pixel_w: int = 1920
pixel_h: int = 1080
# Half size of pixels used frequently
pixel_hw: int = pixel_w // 2
pixel_hh: int = pixel_h // 2

# Path for background frames
path_frames = r'..\frames\Background'


# *************************************************************************************************
def cam2pix(object_xy: np.ndarray):
    """
    Convert the position of an object from 
    (x,y) coordinates in the camera frame, to
    (u,v) pixel locations
    """
    # Reshape the object if necessary so it's an Nx2 matrix
    object_xy = object_xy.reshape((-1,2))
    # Extract x and y vectors from object
    x = object_xy[:, 0]
    y = object_xy[:, 1]
    # Convert these to pixels using the pixel sizes and counts
    u = np.round(+x / pixel_size_x + pixel_hw).astype(np.int16)
    v = np.round(-y / pixel_size_y + pixel_hh).astype(np.int16)
    # Return these as an Nx2 array
    return np.stack([u, v]).T


def make_transforms(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """
    Make two coordinate transform for a camera
    INPUTS:
    cam_pos:   the position of the camera in world coordinates
    cam_point: a point towards which the camera is pointed in world coordinates
    focus:     the focal length of the camera
    zoom:      the zoom setting of the camera
    Returns:
    two transform function that sends (U, V, W) coordinates of an object in the world frame
    to output in the camera frame.
    transform_xy maps to (x, y) coordinates of its image in the camera's focal plane
    transform_pixel maps to (u, v) coordinates of its image in pixel space
    """
    # The effective focus is the manufacturing focus divided by the zoom
    focus_eff = focus / zoom
    
    # Camera z-axis in world coords is the difference between 
    # where the camera is pointing and where it is located
    # z = np.array([-22, 42.5, 2.8]) # 2 worked well
    z = cam_point - cam_pos
    
    # Camera x-axis in world coords
    # x=np.array([42.5,22,0])
    x = np.array([z[1], -z[0], 0.0])
    
    # APPLY TRANSFORMATIONS:
    # Normalize each vector and make a rotation matrix R
    x = x / np.linalg.norm(x)
    z = z / np.linalg.norm(z)
    
    # Get y-axis via cross product
    y = np.cross(x,z)
    # Build rotation matrix and its transpose
    R = np.vstack((x,y,z))
    Rt = np.transpose(R)

    # The transform function to (x, y) coordinates on the camera
    def transform_xy(object_pos):
        """The transform for this camera"""
        # Reshape the object if necessary so it's an Nx3 matrix
        object_pos = object_pos.reshape((-1,3))
        # Compute the position of this object relative to the camera, in the world frame (i.e. coordinates UVW)
        object_UVW = object_pos - cam_pos
        # Apply the rotation Rt to the local object coordinates 
        # to get the position of this object in the camera frame (i.e. coordinates XYZ)
        object_XYZ = np.matmul(object_UVW, Rt)
        # Distance of this object in the Z-plane
        object_Z: float = object_XYZ[:,2]
        # Convert to 2D using focal length and zoom
        scaler = np.reshape(focus_eff / object_Z, (-1, 1))
        object_xy = scaler * object_XYZ[:,0:2]
        # This transform returns (x, y) coordinates of objects
        return object_xy
    
    # The transform function to (u, v) pixel locations
    def transform_uv(object_pos):
        """The transform to pixel space"""
        return cam2pix(transform_xy(object_pos))
    
    # Return the assembled transform
    return transform_xy, transform_uv


def make_transform_uv(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """Wrapper for make_transforms; return only the pixel transform"""
    # Dispatch call to make_transforms to create both xy and pixel transforms
    transform_xy, transform_uv = make_transforms(cam_pos, cam_point, zoom)
    # Return only the pixel transform
    return transform_uv


def annotate_frame(frame_name: str, pixels: np.ndarray, color: str):
    """Annotate a frame with a batch of pixels"""
    pass

# *************************************************************************************************
# All the lines on the floor
court_lines = make_court_lines()
vertical_N = court_lines['vertical_N']
perimeter = court_lines['perimeter']
key_box = court_lines['key_box_N']

# Position of camera (adjusted for height of tripod)
cam_pos = np.array([22,-1.5, 5.2])

# Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
cam_point = np.array([0.0, 41.0, 8.0])

# Transform for camera 3 (to pixel space)
transform = make_transform_uv(cam_pos, cam_point, 1.0)

# Apply the transform for camera 3 to various shapes of interest
# Convert the xy positions to uv pixel locations
key_box_uv = transform(key_box)

# Plot the vertical line under the north basket
vertical_N_uv = transform(vertical_N)
# Plot the perimeter
perimeter_uv = transform(perimeter)

# Demo frame
median_frame = load_frame(path_frames, 'Camera3_median.png')
# Create axes
fig, ax = plt.subplots(figsize=figsize)
ax.set_xlim(0, 1920)
ax.set_ylim(1080, 0)
ax.set_xticks(arange_inc(0, 1920, 120))
ax.set_yticks(arange_inc(0, 1080, 120))
ax.set_xticklabels(arange_inc(0, 192, 12))
ax.set_yticklabels(arange_inc(0, 108, 12))
# Display the image
ax.imshow(median_frame)
# Overlay the pixels
# ax.plot(key_box_uv[:,0], key_box[:,1], color='r')
# ax.plot(origin_uv[:,0], origin_uv[:,1], color='r')
ax.plot(vertical_N_uv[:,0], vertical_N_uv[:,1], color='r')
ax.plot(perimeter_uv[:,0], perimeter_uv[:,1], color='r')
display(fig)
plt.close(fig)


# *************************************************************************************************
