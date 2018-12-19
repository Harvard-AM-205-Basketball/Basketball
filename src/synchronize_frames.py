"""
Harvard IACS Applied Math 205
Project: Basketball

synchronize_frames.py
Extract the background from a sequence of frames taken by the same camera.
Background is inferred as either the mean or median value for each pixel.

Michael S. Emanuel
Mon Dec 17 11:52:58 2018
"""

from shutil import copyfile
from tqdm import tqdm
from am205_utils import range_inc
from image_utils import frame_names
from typing import List, Dict

def main():
    # Path to  frames directory (uncsyncronized)
    path_frames: str = r'../frames'
    # Path to synchronized frames directory
    path_sync_frames: str = r'../sync_frames'
    # List of Camera names
    camera_names: List[str] = [f'Camera{n}' for n in range_inc(8) if n != 5]
    # Number of cameras
    camera_count: int = len(camera_names)
    # Frame lags for each camera (output of audio_synchronize.py)
    camera_lags: Dict[str, int] = {
        'Camera1':   0,
        'Camera2': 222,
        'Camera3': 530,
        'Camera4': 106,
        'Camera6':  45,
        'Camera7': 497,
        'Camera8': 251,
    }
    
    # Largest lag of any camera
    max_lag: int = max(camera_lags.values())
    
    # Iterate over cameras to get all file names for each camera
    # Also compute the number of available frames for each camera after the offset is accounted for
    frame_names_tbl: Dict[str, List[str]] = dict()
    frame_count_tbl: Dict[str, int] = dict()
    offset_tbl: Dict[str, int] = dict()
    for camera_name in camera_names:
        # List of all available frame names for this camera
        frame_names_tbl[camera_name] = frame_names(path_frames, camera_name)
        # Lag for this camera
        camera_lag: int = camera_lags[camera_name]
        # Frame offset applied to this camera so its output will we synchronzied
        offset_tbl[camera_name]  = max_lag - camera_lag
        # The number of available frames for this camera is the total number minus the offset
        frame_count_tbl[camera_name] = len(frame_names_tbl[camera_name]) - offset_tbl[camera_name]
    
    # The number of synchronized frames to output is the minimum over all the cameras
    synch_frame_count: int = min(frame_count_tbl.values())
    
    # Iterate over cameras in a second pass; this time copy frames with synchronized indices
    print(f'Copying synchronized frames for {camera_count} cameras...')
    for camera_name in tqdm(camera_names):
        # Lag for this camera
        camera_lag = camera_lags[camera_name]
        # Frame offset applied to this camera so its output will we synchronzied
        offset: int = offset_tbl[camera_name]
        # Iterate over the synchronized frame numbers
        for i in range(synch_frame_count):
            # The index of the unsynchronized frame corresponding to synchronized frame i
            j = i + offset + 1
            # The frame name (unsynchronized)
            frame_name: str = f'{camera_name}_Frame{j:05d}.png'
            # The frame name (synchronized)
            frame_name_sync: str = f'{camera_name}_SyncFrame{i:05d}.png'
            # Full file names (with path) the the source and target file for copy
            source_name: str = f'{path_frames}/{camera_name}/{frame_name}'
            target_name: str = f'{path_sync_frames}/{camera_name}/{frame_name_sync}'    
            # Copy the frame to the synchronized frame
            copyfile(source_name, target_name)

if __name__ == '__main__':
    main()
