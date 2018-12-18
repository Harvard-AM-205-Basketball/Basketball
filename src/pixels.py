import numpy as np
from numpy import pi
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt
from basketball_court import make_court_lines
from camera_transform import make_transform
from typing import List

# *************************************************************************************************
# Focal length (0.024147 ft is from manufacturer), changes with zoom
focus: float = 0.024147

# All the lines on the floor
court_lines = make_court_lines()
floor_lines = court_lines['floor']
key_box = court_lines['key_box_N']

# *************************************************************************************************
# Position of camera (adjusted for height of tripod)
cam_pos = np.array([22,-1.5, 5.2])

# Camera 3 is ROUGHLY pointed towards a point 2 feet below the front of the rim
cam_point = np.array([0.0, 41.0, 8.0])

# Transform for camera 3
transform = make_transform(cam_pos, cam_point, 1.0)

# Apply the transform for camera 3 to points in the key box
# Orientation is NW corner, NE corner, SE corner, then SW corner. Step size is 0.2
key_box_xy = transform(key_box)

# *************************************************************************************************
# SET UP LEAST SQUARES PROBLEM TO SOLVE FOR PIXEL SIZE
# Coefficient matrix for least squares problem, given image plane coordinates
def coeff_matrix(image_coords):
    A=np.zeros((2*image_coords.shape[0],2))
    for i in range(image_coords.shape[0]):
        A[2*i,0]=image_coords[i,0]
        A[2*i+1,1]=-image_coords[i,1]
    return A

key_coeff = coeff_matrix(key_box_xy)

# The following only works for the key right now but should be generalizable:
# Pixel coordinates of corners of key box, same orientation as in basketball_court (NW, NE, SE, SW)
key_corners_pixel_x=np.array([855,1330,915,290]) 
key_corners_pixel_y=np.array([888,915,1060,1000]) 

# Real-life step size is 0.2
def RHS(pixel_x, pixel_y):
    num_points_x=int(12/0.2+1) # 61 points on free-throw line, baseline
    num_points_y=int(19/0.2+1) # 96 points on other sides
    
    # Pixel x-coords and y-coords of points to be used in least-squares analysis
    xvals=np.concatenate((np.linspace(pixel_x[0],pixel_x[1],num_points_x),np.linspace(pixel_x[1],pixel_x[2],num_points_y),np.linspace(pixel_x[2],pixel_x[3],num_points_x),np.linspace(pixel_x[3],pixel_x[0],num_points_y)))
    yvals=np.concatenate((np.linspace(pixel_y[0],pixel_y[1],num_points_x),np.linspace(pixel_y[1],pixel_y[2],num_points_y),np.linspace(pixel_y[2],pixel_y[3],num_points_x),np.linspace(pixel_y[3],pixel_y[0],num_points_y)))
    
    b=np.zeros(2*len(xvals))
    for i in range(len(xvals)):
        b[2*i]=xvals[i]-960
        b[2*i+1]=yvals[i]-540
    return b

key_RHS = RHS(key_corners_pixel_x,key_corners_pixel_y)
   
# Solve for least-squres pixel size
s=np.linalg.lstsq(key_coeff,key_RHS)[0]
    
# Set pixel width and height based on least squares
pixelheight=1/s[1]
pixelwidth=1/s[0]