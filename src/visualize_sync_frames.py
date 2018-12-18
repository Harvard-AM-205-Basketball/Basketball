"""
Michael S. Emanuel
Tue Dec 18 00:45:43 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from image_background import load_frame
from IPython.display import display
from am205_utils import arange_inc

# Path with synchronized frames
path_frames = r'..\sync_frames'

# Default size for figures to match frames
figsize=[16.0, 9.0]

# Load the frames
frame2 = load_frame(f'{path_frames}/Camera2', 'Camera2_SyncFrame00000.png')
frame3 = load_frame(f'{path_frames}/Camera3', 'Camera3_SyncFrame00000.png')
frame4 = load_frame(f'{path_frames}/Camera4', 'Camera4_SyncFrame00000.png')
frame6 = load_frame(f'{path_frames}/Camera6', 'Camera6_SyncFrame00000.png')
frame7 = load_frame(f'{path_frames}/Camera7', 'Camera7_SyncFrame00000.png')
frame8 = load_frame(f'{path_frames}/Camera8', 'Camera8_SyncFrame00000.png')

frames = [frame2, frame3, frame4, frame6, frame7, frame8]

heights = heights = [9,9,9]
widths = [16, 16]

combined_frame = np.zeros((3*1080, 2*1920, 3))
for i in range(3):
    for j in range(2):
        i0 = 1080*i
        j0 = 1920*j
        combined_frame[i0:i0+1080, j0:j0+1920] = frames[2*i+j]

fig, ax = plt.subplots(figsize=[3*9,2*16])
ax.imshow(combined_frame)
ax.axis('off')
display(fig)
fig.savefig(r'../report/figs/synced_frames.png', bbox_inches='tight')
plt.close(fig)
