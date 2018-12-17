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
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List

# *************************************************************************************************
# Set plot style
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})

# *************************************************************************************************
# Global variables
# Note: units will be in feet not meters because 
# basketball court sizes are standardized in feet in the USA.

# Input constants pertaining to the court
# The width and height of the court in feet
court_w: float = 42.0
court_h: float = 94.0
# The width of the key (rectangle in front of the basket, a.k.a. "restricted area" on Wikipedia diagram)
key_w: float = 12.0
key_h: float = 19.0
# Offset from the perimeter to the backboard
basket_x: float = 4.0
# Height of the basket
basket_h: float = 10.0

# Useful derivated quantities
# The half width and half height
court_hw: float = court_w / 2.0
court_hh: float = court_h / 2.0

# *************************************************************************************************
def make_line(line_beg: np.ndarray, line_end: np.ndarray, step_size: float):
    """Generate an array of 3D sample points connecting the beg to end point."""
    # Length of the line
    line_len: float = np.linalg.norm(line_end - line_beg)
    # Number of points
    point_count: int = int(round(line_len / step_size + 1))
    # Sample points for the line are a convex combination of the begin and end points
    ts = np.linspace(1.0, 0.0, point_count)
    return np.array([t * line_beg + (1.0 - t) * line_end for t in ts])


def make_polygon(points: List[np.ndarray], step_size: float):
    """Generate an array of 3D sample points tracing out the perimater of a polygon."""
    # The number of points in this polygon
    n: int = len(points)
    # Connect point i to point i+1
    lines = [make_line(points[i], points[i+1], step_size) for i in range(n-1)]
    # Connect point n to point 1
    lines += [make_line(points[n-1], points[0], step_size)]
    # Assemble all the lines in the polygon
    return np.vstack(lines)


def make_arc(center: np.ndarray, radius: float, theta_1: float, theta_2: float, num_points: int):
    """Generate an array of 3D sample points on a circle"""
    return center + \
        np.array([radius * np.array([np.cos(theta), np.sin(theta), 0.0]) 
        for theta in np.linspace(theta_1, theta_2, num_points)])

    
def make_circle(center: np.ndarray, radius: float, num_points: int):
    """Generate an array of 3D sample points on a circle"""
    return make_arc(center, radius, 0.0, 2.0 * np.pi, num_points)


# *************************************************************************************************
def make_court_lines():
    """Generate world coordinates of the lines painted red on the basketball court."""
    # Set the step size for line painting
    step_size: float = 0.20
    
    # The main perimeter of the court is just a big rectangle spanning the four corners
    court_NW_x: float = -court_hw
    court_NW_y: float = +court_hh
    court_NE_x: float = +court_hw
    court_NE_y: float = +court_hh
    court_SE_x: float = +court_hw
    court_SE_y: float = -court_hh
    court_SW_x: float = -court_hw
    court_SW_y: float = -court_hh
    
    # Wrap up start and end point at corners of the court into 3D arrays
    court_NW = np.array([court_NW_x, court_NW_y, 0.0])
    court_NE = np.array([court_NE_x, court_NE_y, 0.0])
    court_SE = np.array([court_SE_x, court_SE_y, 0.0])
    court_SW = np.array([court_SW_x, court_SW_y, 0.0])
    
    # Generate the lines around the perimeter of the court
    perimeter = make_polygon([court_NW, court_NE, court_SE, court_SW], step_size)
    
    # The half-court line
    half_court_W = np.array([-court_hw, 0.0, 0.0])
    half_court_E = np.array([+court_hw, 0.0, 0.0])
    half_court = make_line(half_court_W, half_court_E, step_size)
    
    # The center circle is 6 feet in radius
    center_circle = make_circle(np.array([0.0, 0.0, 0.0]), 6.0, 180)
    
    # The "key" in front of the basket has four corners
    # The x coordinates on the left and right are -1/2 to +1/2 of the width of the key
    key_NW_x: float = -key_w / 2.0
    key_NE_x: float = +key_w / 2.0
    key_SW_x: float = -key_w / 2.0
    key_SE_x: float = +key_w / 2.0
    # The y coordinates on the "north" side (top) the key are the half court width
    key_NW_y: float = court_hh
    key_NE_y: float = court_hh
    # The y coordinates on the "south" side (bootom) of the key are 
    # the half court width minus the height of the key
    key_SW_y: float = court_hh - key_h
    key_SE_y: float = court_hh - key_h
    
    # Wrap up start and end point at corners of the key into 3D arrays
    key_NW = np.array([key_NW_x, key_NW_y, 0.0])
    key_NE = np.array([key_NE_x, key_NE_y, 0.0])
    key_SW = np.array([key_SW_x, key_SW_y, 0.0])
    key_SE = np.array([key_SE_x, key_SE_y, 0.0])
    
    # Sample all four lines on the key
    key_box_N = make_polygon([key_NW, key_NE, key_SE, key_SW], step_size)
    # Build "southern" key box by symmetry with "northern" one
    key_box_S = key_box_N.copy()
    key_box_S[:,1] = -key_box_N[:,1]
        
    # The semi-circle on top of the key
    key_circle_radius = key_w / 2.0
    key_circle_center_N = np.array([0.0, +court_hh - key_h, 0.0])
    key_circle_center_S = np.array([0.0, -court_hh + key_h, 0.0])
    key_circle_N = make_circle(key_circle_center_N, key_circle_radius, 360)
    key_circle_S = make_circle(key_circle_center_S, key_circle_radius, 360)
    
    # The three-point arc
    theta = np.pi * (20/360)
    basket_xy_N = np.array([0.0, +court_hh - basket_x, 0.0])
    basket_xy_S = np.array([0.0, -court_hh + basket_x, 0.0])
    three_point_N = make_arc(basket_xy_N, 21.0, -np.pi + theta, -theta, 360)
    three_point_S = make_arc(basket_xy_S, 21.0, theta, pi - theta, 360)
    
    # Vertical line under the basket
    vertical_to_basket = np.array([0.0, 0.0, basket_h])
    vertical_N = make_line(basket_xy_N, basket_xy_N + vertical_to_basket, step_size)
    vertical_S = make_line(basket_xy_S, basket_xy_S + vertical_to_basket, step_size)
    
    # The small white square in the backboard has 
    # https://www.google.com/imgres?imgurl=https://i.pinimg.com/originals/37/15/f7/3715f7d7d509c0ff9b8f9f6d3dd5f15d.jpg&imgrefurl=https://www.pinterest.com/pin/263531015675013403/&h=462&w=500&tbnid=VDOxYRz-f96HcM:&q=basketball+backboard+dimensions&tbnh=160&tbnw=173&usg=AI4_-kR6HhW5heRUfNGcRcWqbUhvRuguPg&vet=12ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg..i&docid=3HCf0A11ry3g-M&sa=X&ved=2ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg#h=462&imgdii=kW4tKWYY4k45EM:&tbnh=160&tbnw=173&vet=12ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg..i&w=500
    # x coordinates +/- 1
    # y coordaintes 43 (47 feet half court minus 4 feet forward)
    # z coordinates 10 to 11.5 and 
    bb_sq_BL = np.array([-1.0, court_hh - basket_x, basket_h])
    bb_sq_BR = np.array([+1.0, court_hh - basket_x, basket_h])
    bb_sq_TL = np.array([-1.0, court_hh - basket_x, basket_h + 1.5])
    bb_sq_TR = np.array([+1.0, court_hh - basket_x, basket_h + 1.5])
    bb_sq = make_polygon([bb_sq_BL, bb_sq_TL, bb_sq_TR, bb_sq_BR], 0.25)

    # All the lines on the floor
    floor = np.vstack([perimeter, half_court, center_circle, key_box_N, key_box_S,
                       key_circle_N, key_circle_S, three_point_N, three_point_S])

    # Wrap these up into a dict
    lines = dict()
    lines['floor'] = floor
    lines['perimeter'] = perimeter
    lines['half_court'] = half_court
    lines['center_circle'] = center_circle
    lines['key_box_N'] = key_box_N
    lines['key_box_S'] = key_box_S
    lines['key_circle_N'] = key_circle_N
    lines['key_circle_S'] = key_circle_S
    lines['three_point_N'] = three_point_N
    lines['three_point_S'] = three_point_S
    lines['vertical_N'] = vertical_N
    lines['vertical_S'] = vertical_S
    lines['bb_sq'] = bb_sq

    return lines


# *************************************************************************************************
def visualize_court(lines):
    """Visualize the lines on a basketball court"""
    # Unpack lines
    perimeter = lines['perimeter']
    half_court = lines['half_court']
    center_circle = lines['center_circle']

    key_box_N = lines['key_box_N']
    key_circle_N = lines['key_circle_N']

    key_box_S = lines['key_box_S']
    key_circle_S = lines['key_circle_S']

    three_point_N = lines['three_point_N']
    three_point_S = lines['three_point_S']

    # Plot the lines
    fig, ax = plt.subplots(figsize=[court_w / 8, court_h / 8])
    ax.set_title('Basketball Court')
    ax.set_xlim(-court_hw-1, court_hw+1)
    ax.set_ylim(-court_hh-1, court_hh+1)
    linewidth=0
    marker='o'
    markersize=1.0
    
    ax.plot(perimeter[:,0], perimeter[:,1], color='r', label='perimeter', 
            linewidth=linewidth, marker=marker, markersize=markersize)
    ax.plot(half_court[:,0], half_court[:,1], color='r', label='halfcourt',
            linewidth=linewidth, marker=marker, markersize=markersize)
    ax.plot(center_circle[:,0], center_circle[:,1], color='r', label='center_circle',
            linewidth=linewidth, marker=marker, markersize=markersize)            
    ax.plot(key_box_N[:,0], key_box_N[:,1], color='r', label='key',
            linewidth=linewidth, marker=marker, markersize=markersize)            
    ax.plot(key_circle_N[:,0], key_circle_N[:,1], color='r', label='key',
            linewidth=linewidth, marker=marker, markersize=markersize)

    ax.plot(key_box_S[:,0], key_box_S[:,1], color='r', label='key',
            linewidth=linewidth, marker=marker, markersize=markersize)            
    ax.plot(key_circle_S[:,0], key_circle_S[:,1], color='r', label='key',
            linewidth=linewidth, marker=marker, markersize=markersize)

    ax.plot(three_point_N[:,0], three_point_N[:,1], color='r', label='3pt',
            linewidth=linewidth, marker=marker, markersize=markersize)            
    ax.plot(three_point_S[:,0], three_point_S[:,1], color='r', label='3pt',
            linewidth=linewidth, marker=marker, markersize=markersize)            

# *************************************************************************************************
# main
def main():
    # The lines on the court
    lines = make_court_lines()
    
    # Visualize the lines on the court
    visualize_court(lines)

if __name__ == '__main__':
    main()
