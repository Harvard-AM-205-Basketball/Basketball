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
from IPython.display import display
from am205_utils import arange_inc
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
# Note: these are the dimensions of the MAC at Harvard, which are HIGHLY NONSTANDARD!
court_w: float = 40.0
court_h: float = 88.0
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
    # https://en.wikipedia.org/wiki/Basketball_court
    # https://www.sportsknowhow.com/basketball/dimensions/high-school-basketball-court-dimensions.html
    
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
    key_circle_N = make_arc(key_circle_center_N, key_circle_radius, 1*pi, 2*pi, 360)
    key_circle_S = make_arc(key_circle_center_S, key_circle_radius, 0*pi, 1*pi, 360)

    # The y coordinates of both baskets; they are 43 feet from the center
    # (the height of a half court is 47 feet, and the basket is 4 feet inside the back line)
    basket_y_N: float = +court_hh - basket_x
    basket_y_S: float = -court_hh + basket_x
    
    # The 3D coordinates of the basket (where the rim backplate is mounted to the backboard)
    basket_N = np.array([0.0, basket_y_N, basket_h])
    basket_S = np.array([0.0, basket_y_S, basket_h])

    # The point underneath the basket on the floor
    basket_xy_N = np.array([0.0, basket_y_N, 0.0])
    basket_xy_S = np.array([0.0, basket_y_S, 0.0])

    # 3D vector with discplacement from floor to basket
    # vertical_to_basket = np.array([0.0, 0.0, basket_h])
    
    # The three-point arc
    theta = np.pi * (36/360)
    three_point_N = make_arc(basket_xy_N, 21.0, -np.pi + theta, -theta, 360)
    three_point_S = make_arc(basket_xy_S, 21.0, theta, pi - theta, 360)
    
    # Vertical line under the basket    
    vertical_N = make_line(basket_xy_N, basket_N, step_size)
    vertical_S = make_line(basket_xy_S, basket_S, step_size)

    # See this image for the layout of the backboard
    # https://en.wikipedia.org/wiki/Backboard_(basketball)
    # https://www.google.com/imgres?imgurl=https://i.pinimg.com/originals/37/15/f7/3715f7d7d509c0ff9b8f9f6d3dd5f15d.jpg&imgrefurl=https://www.pinterest.com/pin/263531015675013403/&h=462&w=500&tbnid=VDOxYRz-f96HcM:&q=basketball+backboard+dimensions&tbnh=160&tbnw=173&usg=AI4_-kR6HhW5heRUfNGcRcWqbUhvRuguPg&vet=12ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg..i&docid=3HCf0A11ry3g-M&sa=X&ved=2ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg#h=462&imgdii=kW4tKWYY4k45EM:&tbnh=160&tbnw=173&vet=12ahUKEwjG1t7_16ffAhXuY98KHSVDDm0Q9QEwAHoECAQQBg..i&w=500
    
    # The large white square in the backboard has
    # x coordinates +/- 3 feet (it is 72 inches wide and symmetrical)
    # y coordinates 43 (47 feet half court minus 4 feet forward)
    # z coordinates 9.5 to 13.0 feet; it is 42 inches high and starts 6 inches under the rim)
    backboard_BL = np.array([-3.0, basket_y_N, basket_h - 0.5])
    backboard_BR = np.array([+3.0, basket_y_N, basket_h - 0.5])
    backboard_TL = np.array([-3.0, basket_y_N, basket_h + 3.0])
    backboard_TR = np.array([+3.0, basket_y_N, basket_h + 3.0])
    # Wrap up the four corners into a rectangle
    backboard_N = make_polygon([backboard_BL, backboard_TL, backboard_TR, backboard_BR], 0.25)
    # Build "southern" backboard square backboard_in by symmetry with "northern" one
    backboard_S = backboard_N.copy()
    backboard_S[:,1] = -backboard_N[:,1]    
    
    # The small white square in the backboard has 
    # x coordinates +/- 1 foot (it is 24 inches wide and symmetrical)
    # y coordinates 43 (47 feet half court minus 4 feet forward)
    # z coordinates 10 to 11.5 (the rim is 10.0 feet high and the box is 18 inches tall)
    backboard_in_BL = np.array([-1.0, basket_y_N, basket_h])
    backboard_in_BR = np.array([+1.0, basket_y_N, basket_h])
    backboard_in_TL = np.array([-1.0, basket_y_N, basket_h + 1.5])
    backboard_in_TR = np.array([+1.0, basket_y_N, basket_h + 1.5])
    # Wrap up the four corners into a rectangle
    backboard_in_N = make_polygon([backboard_in_BL, backboard_in_TL, backboard_in_TR, backboard_in_BR], 0.25)
    # Build "southern" backboard square backboard_in by symmetry with "northern" one
    backboard_in_S = backboard_in_N.copy()
    backboard_in_S[:,1] = -backboard_in_N[:,1]    

    # The rim
    # https://www.livestrong.com/article/405043-basketball-rim-measurements/
    # A basketball hoop is circular in shape and has an inside diameter of exactly 18 inches from edge to edge. 
    # There should be 6 inches of separation from the ring to the basket. 

    # The diameter is 18 inches so the radius is 9 inches = 0.75 feet
    rim_radius = 0.75
    
    # x coordinate of center: 0
    # y coordinate of center: 42.5 feet = basket_y_N - 1.25: 
    # the offset to center is: 6 inches offset + 9 inch radius = 15 inches = 1.25 feet
    # z coordinate: 10.0 feet = basket_h
    rim_center_N = np.array([0.0, basket_y_N - 1.25, basket_h])
    rim_center_S = np.array([0.0, basket_y_S + 1.25, basket_h])
    rim_N = make_circle(rim_center_N, rim_radius, 180)
    rim_S = make_circle(rim_center_S, rim_radius, 180)

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
    lines['backboard_N'] = backboard_N
    lines['backboard_S'] = backboard_S
    lines['backboard_in_N'] = backboard_in_N
    lines['backboard_in_S'] = backboard_in_S
    lines['rim_N'] = rim_N
    lines['rim_S'] = rim_S

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
    
    rim_N = lines['rim_N']
    rim_S = lines['rim_S']
    
    backboard_N = lines['backboard_N']
    backboard_S = lines['backboard_S']

    # Plot the lines
    fig, ax = plt.subplots(figsize=[court_w / 8, court_h / 8])
    ax.set_title('MAC Basketball Court')
    ax.set_xlim(-court_hw-1, court_hw+1)
    ax.set_ylim(-court_hh-1, court_hh+1)
    ax.set_xticks(arange_inc(-20, 20, 2))
    ax.set_yticks(arange_inc(-44, 44, 2))
    xticklabels = [f'{x:d}' if int(x) in [-20, 0, 20] else '' for x in ax.get_xticks()]
    yticklabels = [f'{y:d}' if int(y) in [-44, 0, 44] else '' for y in ax.get_yticks()]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.grid()
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

    ax.plot(rim_N[:,0], rim_N[:,1], color='darkorange', label='rim',
            linewidth=linewidth, marker=marker, markersize=markersize)
    ax.plot(rim_S[:,0], rim_S[:,1], color='darkorange', label='rim',
            linewidth=linewidth, marker=marker, markersize=markersize)

    ax.plot(backboard_N[:,0], backboard_N[:,1], color='k', label='rim',
            linewidth=4, marker=marker, markersize=markersize)
    ax.plot(backboard_S[:,0], backboard_S[:,1], color='k', label='rim',
            linewidth=4, marker=marker, markersize=markersize)


    
    # Add the 8 cameras
    ax.plot(0, +court_hh, 'o', color='b', markersize=10)
    ax.plot(+court_hw, +court_hh, 'o', color='b', markersize=10)
    ax.plot(+court_hw, 0, 'o', color='b', markersize=10)
    ax.plot(+court_hw, -court_hh, 'o', color='b', markersize=10)
    ax.plot(0, -court_hh, 'o', color='b', markersize=10)
    ax.plot(-court_hw, -court_hh, 'o', color='b', markersize=10)
    ax.plot(-court_hw, 0, 'o', color='b', markersize=10)
    ax.plot(-court_hw, court_hh, 'o', color='b', markersize=10)
    
    # Save the figure
    display(fig)
    fig.savefig('../report/figs/court_lines.png', bbox_inches='tight')
    plt.close(fig)

# *************************************************************************************************
# main
def main():
    # The lines on the court
    lines = make_court_lines()
    
    # Visualize the lines on the court
    visualize_court(lines)

if __name__ == '__main__':
    main()
