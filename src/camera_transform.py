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
from numpy import pi
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt
# from am205_utils import arange_inc
from basketball_court import make_court_lines
from typing import List

# *************************************************************************************************
# Set plot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})

# Default size for figures to match frames
figsize=[19.2, 10.8]

# Focal length (0.024147 ft is from manufacturer), changes with zoom
focus: float = 0.024147

# All the lines on the floor
court_lines = make_court_lines()
floor_lines = court_lines['floor']
key_box = court_lines['key_box_N']

# *************************************************************************************************

def make_transform(cam_pos: np.ndarray, cam_point: np.ndarray, zoom: float):
    """
    Make a coordinate transform for a camera
    INPUTS:
    cam_pos:   the position of the camera in world coordinates
    cam_point: a point towards which the camera is pointed in world coordinates
    focus:     the focal length of the camera
    zoom:      the zoom setting of the camera
    Returns:
    a transform function that sends (U, V, W) coordinates of an object in the world frame
    to (x, y) coordinates of its image in the camera's focal plane (not pixels!)
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

    # The transform function
    def transform(object_pos):
        """The transform for this camera"""
        # Compute the position of this object relative to the camera, in the world frame (i.e. coordinates UVW)
        object_UVW = object_pos - cam_pos
        # Apply the rotation Rt to the local object coordinates 
        # to get the position of this object in the camera frame (i.e. coordinates XYZ)
        object_XYZ = np.matmul(object_UVW, Rt)
        # Distance of this object in the Z-plane
        object_Z: float = object_XYZ[2]
        # Convert to 2D using focal length and zoom
        object_xy = (focus_eff / object_Z) * object_XYZ[0:2]
        # This transform returns (x, y) coordinates of objects
        return object_xy
    
    # Return the assembled transform
    return transform

# Position of camera (adjusted for height of tripod)
cam_pos = np.array([22,-1.5, 5.2])

# Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
cam_point = np.array([0.0, 41.0, 8.0])

# Transform for camera 3
# transform = make_transform(cam_pos, cam_point, focus, 1.0)

# Apply the transform to points in the key box


zoom = 1.0
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
y = np.cross(x, z)
# Build rotation matrix and its transpose
R = np.vstack((x, y, z))
Rt = np.transpose(R)

# The transform function
def transform(object_pos):
    """The transform for this camera"""
    # Compute the position of this object relative to the camera, in the world frame (i.e. coordinates UVW)
    object_UVW = object_pos - cam_pos
    # Apply the rotation Rt to the local object coordinates 
    # to get the position of this object in the camera frame (i.e. coordinates XYZ)
    object_XYZ = np.matmul(object_UVW, Rt)
    # Distance of this object in the Z-plane
    object_Z: float = object_XYZ[2]
    # Convert to 2D using focal length and zoom
    object_xy = (focus / object_Z) * object_XYZ[0:2]
    # This transform returns (x, y) coordinates of objects
    return object_xy



#    # Subtract position of camera on tripod
#    coords = coords - campos
#    
#    # Apply rotation matrix (easier to use R^T here)
#    coords=np.matmul(coords, Rt)
#    
#    # Convert to 2D by using focal length
#    coords[:,0]=f*coords[:,0]/coords[:,2] # Divide X by Z and multiply by f
#    coords[:,1]=f*coords[:,1]/coords[:,2] # Divide Y by Z and multiply by f
#    
#    
#    # Delete z-coordinate (not needed since pixel plane is 2D)
#    coords = np.delete(coords, 2, 1)
