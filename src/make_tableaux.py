"""
Harvard IACS Applied Math 205
Project: Basketball

make_tableaux:
Batch process to generate and save tableau images.
A tableau is one big image consolidating 6 frames for cameras 2, 3, 4, 6, 7, 8.
Each tableau has pixel size
height = 1080x3 = 3240
width  = 1920x2 = 3840 

Michael S. Emanuel
Wed Dec 19 16:16:10 2018
"""

import sys
from textwrap import dedent
from joblib import Parallel, delayed
from tqdm import tqdm
from image_utils import make_tableau, plot_tableau
from typing import List

# *************************************************************************************************
# Path to tableau
path_tableau: str = '../tableau'

# Number of frames
frame_count: int = 4391

# Number of threads
jobs: int = 16

def process_frames(frame_nums, progress_bar: bool = False):
    """Process a batch of frames"""
    # Wrap the frame_nums in tqdm if progress_bar was specified
    frame_iter = tqdm(frame_nums) if progress_bar else frame_nums
    # Iterate over the frame numbers
    for n in frame_iter:
        frames = make_tableau(n)
        fig, ax = plot_tableau(frames)
        fig.savefig(f'{path_tableau}/Tableau{n:05d}.png')


def main():
    # Range of frames to process
    n0: int
    n1: int

    # Process command line arguments
    argv: List[str] = sys.argv
    argc: int = len(sys.argv)-1
    usage_str = dedent(
    """
    python make_tableaux.py 
        process all frames
    python make_tableaux.py n0 n1
        calculate background for cameras numbered [c0, ..., c1)
    """)
    try:
        if argc == 0:
            n0 = 0
            n1 = frame_count
        elif argc == 1:
            raise RuntimeError
        elif argc == 2:
            n0 = int(argv[1])
            n1 = int(argv[2])
        else:
            raise RuntimeError
    except:
        print(f'Error in arguments for make_tableaux.py.  argc={argc}, argv={argv}.')
        print(usage_str)
        exit()
    print(f'Processing frames from {n0} to {n1} on {jobs} threads.')
    
    # Split up the frames for apportionment to different threads
    frame_nums = list(range(n0, n1))
    job_tbl = dict()
    for k in range(jobs):
        job_tbl[k] = [n for n in frame_nums if n % jobs == k]
        
    # List of arguments for parallel job
    args = [(job_tbl[jn], jn == 1) for jn in range(jobs)]
    
    # Run these jobs in parallel
    Parallel(n_jobs=jobs)(
            delayed(process_frames)(frame_nums, progress_bar)
            for frame_nums, progress_bar in args)

if __name__ == '__main__':
    main()
