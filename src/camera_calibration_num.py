"""
Harvard IACS Applied Math 205
Project: Basketball

camera_calibration.py
infer the location and orientation of cameras by numerical calibration.

Michael S. Emanuel
Wed Dec 19 18:36:05 2018
"""

import numpy as np
from scipy.optimize import minimize
from basketball_court import make_court_landmarks
from camera_transform import make_transforms_angle, sph_from_points
from camera_calibration import transforms as transforms_cal, cam_pos_tbl, cam_point_tbl, zoom_tbl
from camera_calibration import calibrate_cam1, calibrate_cam2, calibrate_cam3, calibrate_cam4
from camera_calibration import calibrate_cam6, calibrate_cam7, calibrate_cam8
from am205_utils import range_inc
from typing import List

# *************************************************************************************************

# Generate the landmarks
landmarks_court = make_court_landmarks()
# Table of landmarks by camera
landmarks_tbl = dict()

# Camera numbers
camera_nums: np.ndarray = np.array([i for i in range_inc(8) if i != 5], dtype=np.int8)
# Camera names
camera_names: List[str] = [f'Camera{i}' for i in camera_nums]

# Get the pixel transforms for all the cameras from the visual calibration
# transforms_vis[camera_name] is transform_fg for the named camera; 
# maps position to floating point pixel (u, v)
transforms_vis = dict()
# transforms_xy = np.empty(7, dtype=object)
for camera_name in camera_names:
    transforms_vis[camera_name] = transforms_cal[camera_name][2]

# Visual calibration function for all cameras
visual_cal_func_tbl = dict()
visual_cal_func_tbl['Camera1'] = calibrate_cam1
visual_cal_func_tbl['Camera2'] = calibrate_cam2
visual_cal_func_tbl['Camera3'] = calibrate_cam3
visual_cal_func_tbl['Camera4'] = calibrate_cam4
visual_cal_func_tbl['Camera6'] = calibrate_cam6
visual_cal_func_tbl['Camera7'] = calibrate_cam7
visual_cal_func_tbl['Camera8'] = calibrate_cam8


# *************************************************************************************************
# Optimization objective function
def make_objective_func(landmarks_xyz, landmarks_uv):
    """Make the optimization objective function"""
    
    def score_func(transform_fg):
        """Compute the score function in terms of a transform"""
        # Apply them to the landmarks        
        calc_uv = transform_fg(landmarks_xyz)
        # The error by pixel
        err = calc_uv - landmarks_uv
        # We want to minimize the sum of squares (the score)
        return np.sum(err * err)

    def param_mapper(params: np.ndarray):
        """Mapper to unpack parameters"""
        # First the camera position
        cam_pos: np.ndarray = params[0:3]
        # Then two angles determining the aim point
        theta: float = params[3]
        phi: float = params[4]
        # The zoom
        zoom: float = params[5]
        return cam_pos, theta, phi, zoom
        
    def objective_func(params: np.ndarray):
        """Compute the error to landmarks using these """
        # Unpack the input arguments
        cam_pos, theta, phi, zoom = param_mapper(params)
        # Build transforms with these parameter values
        transforms = make_transforms_angle(cam_pos, theta, phi, zoom)
        # Extract the fg transformer from these transforms
        transform_fg = transforms[2]
        # Compute the score
        return score_func(transform_fg)

    # Return the assembled objective function
    return objective_func, score_func, param_mapper


# *************************************************************************************************
def calibrate_camera(camera_num: int):
    """Calibrate one camera based on known landmarks"""
    # Camera name
    camera_name: str = f'Camera{camera_num}'
    # Look up the landmarks for this camera on the big landmarks table
    landmarks = landmarks_tbl[camera_name]
    
    # Ordered list of these landmarks
    landmark_names = [nm for nm in landmarks.keys()]
    # Number of landmarks
    landmark_count: int = len(landmark_names)
    
    # Visual calibration function for this camera
    visual_cal_func = visual_cal_func_tbl[camera_name]
    
    # Convert landmarks into two matrices: one with the world coordinates, one with pixel locations
    landmarks_xyz = np.array([landmarks_court[nm] for nm in landmark_names])
    landmarks_uv = np.array([landmarks[nm] for nm in landmark_names])
    
    # Make objective for this camera
    objective_func, score_func, param_mapper = make_objective_func(landmarks_xyz, landmarks_uv)

    # Current visual calibration
    transform_vis = transforms_vis[camera_name]
    
    # Apply this to landmarks (for interactive testing)
    # calc_uv_vis = transform_vis(landmarks_xyz)    
    # Compare to predicted results
    # err_vis = calc_uv_vis - landmarks_uv
    
    # Compute the score and RMS pixel error
    score_vis = score_func(transform_vis)
    rms_vis = np.sqrt(score_vis / (landmark_count*2))
    
    # Get the spherical coordinates corresponding to the visual calibration
    cam_pos_vis = cam_pos_tbl[camera_name]
    cam_point_vis = cam_point_tbl[camera_name]
    zoom_vis = zoom_tbl[camera_name]
    theta_vis, phi_vis = sph_from_points(cam_pos_vis, cam_point_vis)
    
    # Initial guess built from angles
    transforms_num = make_transforms_angle(cam_pos_vis, theta_vis, phi_vis, zoom_vis)
    transform_xy, transform_uv, transform_fg = transforms_num    
    # Pack the initial guess into a params array
    x0 = np.hstack([cam_pos_vis, theta_vis, phi_vis, zoom_vis])

    # Run the optimization
    res = minimize(fun=objective_func, x0=x0)
    params = res['x']

    # Unpack the parameters
    cam_pos, theta, phi, zoom = param_mapper(params)
    # Generate numerical transforms
    transforms = make_transforms_angle(cam_pos, theta, phi, zoom)

    # Final score after optimization
    score_num = objective_func(x0)
    rms_num = np.sqrt(score_num / (landmark_count*2))

    # Run calibration on the numerical transforms
    visual_cal_func(transforms, 'numerical')

    # Report results
    print(f'Results for calibration of {camera_name}:')
    print(f'Visual calibration:    score {score_vis:0.2f} / RMS pixel error {rms_vis:0.3f}.')
    print(f'Numerical calibration: score {score_num:0.2f} / RMS pixel error {rms_num:0.3f}.')
    print(f'Old position: {np.round(cam_pos_vis, 2)}')
    print(f'New position: {np.round(cam_pos, 2)}')
    print(f'Theta: {theta_vis:5.3f} --> {theta:5.3f}')
    print(f'Phi:   {phi_vis:5.3f} --> {theta:5.3f}')
    print(f'Zoom: {zoom_vis:5.3f} --> {zoom:5.3f}')
        
    # Return the transforms and the parmeters
    return transforms, cam_pos, theta, phi, zoom


# *************************************************************************************************
# All the landmarks
# court_NW, court_NE, court_SE, court_SW,
# half_court_W, half_court_E,
# center, center_W,center_E,
# key_N_NW, key_N_NE, key_N_SW, key_N_SE,
# key_S_NW, key_S_NE, key_S_SW, key_S_SE,
# key_top_N, key_top_S,
# backboard_N_BL, backboard_N_BR, backboard_N_TL, backboard_N_TR,
# backboard_S_BL, backboard_S_BR, backboard_S_TL, backboard_S_TR, 
# backboard_in_N_BL, backboard_in_N_BR, backboard_in_N_TL, backboard_in_N_TR,
# backboard_in_S_BL, backboard_in_S_BR, backboard_in_S_TL, backboard_in_S_TR,
# rim_N, rim_S

# *************************************************************************************************
# Camera 1 landmarks
landmarks = dict()
# South corners
landmarks['court_SE'] = np.array([544, 722])
landmarks['court_SW'] = np.array([1447, 781])
# Center
landmarks['center'] = np.array([977, 1029])
landmarks['center_E'] = np.array([688, 1009])
landmarks['center_W'] = np.array([1268, 1046])
landmarks['center_S'] = np.array([982, 960])

# Key box
landmarks['key_S_NE'] = np.array([812, 822])
landmarks['key_S_NW'] = np.array([1164, 839])
landmarks['key_S_SE'] = np.array([854, 752])
landmarks['key_S_SW'] = np.array([1127, 767])

# Save these landmarks
landmarks_tbl['Camera1'] = landmarks


# *************************************************************************************************
# Camera 3 landmarks
landmarks = dict()
# NW corner
landmarks['court_NW'] = np.array([426,861])
# Key box
landmarks['key_N_NW'] = np.array([854, 886])
landmarks['key_N_NE'] = np.array([1327, 915])
landmarks['key_N_SW'] = np.array([287, 997])
landmarks['key_N_SE'] = np.array([914, 1061])
# Backboard (outer)
landmarks['backboard_N_BL'] = np.array([880, 474])
landmarks['backboard_N_BR'] = np.array([1115, 459])
landmarks['backboard_N_TL'] = np.array([879, 330])
landmarks['backboard_N_TR'] = np.array([1116, 308])
# Backboard(inner)
landmarks['backboard_in_N_BL'] = np.array([956, 454])
landmarks['backboard_in_N_BR'] = np.array([1031, 456])
landmarks['backboard_in_N_TL'] = np.array([954, 398])
landmarks['backboard_in_N_TR'] = np.array([1032, 392])

# Save these landmarks
landmarks_tbl['Camera3'] = landmarks


# *************************************************************************************************
# Camera 4 landmarks
landmarks = dict()
# NW corner
landmarks['court_NW'] = np.array([61, 771])
# Center
landmarks['center'] = np.array([144, 1040])
landmarks['center_E'] = np.array([613, 1080])
landmarks['center_N'] = np.array([301, 980])
# Key box
landmarks['key_N_SE'] = np.array([1084, 866])
landmarks['key_N_SW'] = np.array([388, 849])
landmarks['key_N_NW'] = np.array([686, 782])
landmarks['key_N_NE'] = np.array([1273, 789])
landmarks['key_N_top'] = np.array([607, 890])
# Backboard (outer)
landmarks['backboard_N_TL'] = np.array([777, 121])
landmarks['backboard_N_TR'] = np.array([1070, 110])
landmarks['backboard_N_BL'] = np.array([776, 296])
landmarks['backboard_N_BR'] = np.array([1072, 284])
# Backboard(inner)
landmarks['backboard_in_N_TL'] = np.array([878, 203])
landmarks['backboard_in_N_TR'] = np.array([969, 201])
landmarks['backboard_in_N_BL'] = np.array([877, 275])
landmarks['backboard_in_N_BR'] = np.array([971, 268])

# Save these landmarks
landmarks_tbl['Camera4'] = landmarks


# *************************************************************************************************
# Camera 6 landmarks
landmarks = dict()
# NW corner
# landmarks['court_NE'] = np.array([])
# Key box
landmarks['key_N_top'] = np.array([1323, 938])
landmarks['key_N_SE'] = np.array([1558, 907])
landmarks['key_N_SW'] = np.array([781, 919])
landmarks['key_N_NE'] = np.array([1199, 831])
landmarks['key_N_NW'] = np.array([572, 837])
# Backboard (outer)
landmarks['backboard_N_TL'] = np.array([772, 89])
landmarks['backboard_N_TR'] = np.array([1100, 98])
landmarks['backboard_N_BL'] = np.array([774, 288])
landmarks['backboard_N_BR'] = np.array([1097, 288])
# Backboard(inner)
landmarks['backboard_in_N_TL'] = np.array([885, 194])
landmarks['backboard_in_N_TR'] = np.array([987, 196])
landmarks['backboard_in_N_BL'] = np.array([885, 270])
landmarks['backboard_in_N_BR'] = np.array([990, 275])

# Save these landmarks
landmarks_tbl['Camera6'] = landmarks

# *************************************************************************************************
# Calibrate the cameras
transforms, cam_pos, theta, phi, zoom = calibrate_camera(1)
transforms, cam_pos, theta, phi, zoom = calibrate_camera(3)
transforms, cam_pos, theta, phi, zoom = calibrate_camera(4)
transforms, cam_pos, theta, phi, zoom = calibrate_camera(6)
