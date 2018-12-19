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
from numpy import sqrt, sin, cos, arccos, arctan2

# *************************************************************************************************
# Focal length (0.024147 ft is from manufacturer), changes with zoom
focus: float = 0.024147

# Pixel size - obtained by running pixel.py
pixel_size_x: float = 1.196e-5
pixel_size_y: float = 1.141e-5
# Mean pixel size
pixel_size: float = 1.166E-5

# Pixel count in the image
pixel_w: int = 1920
pixel_h: int = 1080

# Half size of pixels used frequently
pixel_hw: int = pixel_w // 2
pixel_hh: int = pixel_h // 2


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


def cam2pix_f(object_xy: np.ndarray):
    """
    Convert the position of an object from 
    (x,y) coordinates in the camera frame, to
    (u,v) pixel locations
    This version is UNROUNDED, i.e. u and v are both floats
    This makes it differentiable and suitable for numerical methods.
    """
    # Reshape the object if necessary so it's an Nx2 matrix
    object_xy = object_xy.reshape((-1,2))
    # Extract x and y vectors from object
    x = object_xy[:, 0]
    y = object_xy[:, 1]
    # Convert these to pixels using the pixel sizes and counts
    u = +x / pixel_size_x + pixel_hw
    v = -y / pixel_size_y + pixel_hh
    # Return these as an Nx2 array
    return np.stack([u, v]).T


# *************************************************************************************************
def sph2xyz(theta: float, phi: float):
    """Convert the polar angles theta and phi to normalized x, y, z coordinates"""
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # use the Mathematics convention because AM-205 has "math" in the course name
    # theta is the angle in the (x, y) plane (same as for 2D)
    # phi is the angle down from the "north pole" betwwen 0 and pi
    
    # The height z
    z = cos(phi)
    # The distance in the XY plane
    r_xy = sin(phi)
    # The x and y coordinates are as usual
    x = r_xy * cos(theta)
    y = r_xy * sin(theta)
    # Assemble into an Nx3 matrix
    return np.stack([x, y, z]).T


def xyz2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Convert Cartesian coordinates x, y, z to polar angles theta and phi"""
    # The size of each vector
    r = sqrt(x*x + y*y + z*z)
    # The angle phi from the north pole (modified latitude)
    phi = arccos(z / r)
    # The angle theta only depends on x and y
    theta = arctan2(y / r, x / r)    
    # Return angles as a pair of arrays
    return theta, phi


# *************************************************************************************************
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
    # The effective focus is the manufacturing focus multiplied by the zoom
    focus_eff = focus * zoom
    
    # Camera z-axis in world coords is the difference between 
    # where the camera is pointing and where it is located
    z = cam_point - cam_pos
    
    # Camera x-axis in world coords
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
    
    # The transform function to (u, v) pixel locations
    def transform_fg(object_pos):
        """The transform to pixel space"""
        return cam2pix_f(transform_xy(object_pos))
    
    # Return the assembled transform
    return transform_xy, transform_uv, transform_fg


# *************************************************************************************************
def make_transform_xy(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """Wrapper for make_transforms; return only the pixel transform"""
    # Dispatch call to make_transforms to create both xy and pixel transforms
    transform_xy, transform_uv, transform_fg = make_transforms(cam_pos, cam_point, zoom)
    # Return only the xy transform
    return transform_xy


def make_transform_uv(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """Wrapper for make_transforms; return only the pixel transform"""
    # Dispatch call to make_transforms to create both xy and pixel transforms
    transform_xy, transform_uv, transform_fg = make_transforms(cam_pos, cam_point, zoom)
    # Return only the pixel transform
    return transform_uv


def make_transform_fg(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """Wrapper for make_transforms; return only the pixel transform"""
    # Dispatch call to make_transforms to create both xy and pixel transforms
    transform_xy, transform_uv, transform_fg = make_transforms(cam_pos, cam_point, zoom)
    # Return only the UNROUNDED (floating point) pixel transform
    return transform_fg
