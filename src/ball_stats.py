"""
Harvard IACS Applied Math 205
Project: Basketball

ball_stats.py: Assemble csv files into one dataframe with the ball position
at frames where it was estimated.

Michael S. Emanuel
Thu Dec 20 08:57:03 2018
"""

import glob
import numpy as np
import pandas as pd
from typing import List

# Pathname for calculation results, e.g. ../calculations/ball_pos_01234.csv
pathname: str = '../calculations/ball_pos*.csv'
# All file names with ball statistics
fnames: List[str] = glob.glob(pathname)

def main():
    # Create a list of dictionaries
    # See e.g. https://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
    row_list = list()
    for fname in fnames:
        df = pd.read_csv(fname)
        entry = df.iloc[0]
        row = {'n': int(entry.n),
               't': entry.t,
               'x': entry.x,
               'y': entry.y,
               'z': entry.z,
               'mask': int(entry['mask'])}
        row_list.append(row)
    
    # Assemble the rows into a DataFrame
    ball_pos_new = pd.DataFrame(row_list)
    # Set the frame number, n, as the index
    ball_pos_new = ball_pos_new.set_index(['n'])
    
    # Name of the ball position CSV file
    fname_df = '../calculations/ball_pos.csv'
    
    # If there is already a ball_pos present, load it and update the contents
    # https://stackoverflow.com/questions/33001585/pandas-dataframe-concat-update-upsert
    try:
        ball_pos_old = pd.read_csv(fname_df, index_col=['n'])
        df1 = ball_pos_old
        df2 = ball_pos_new
        # mask for non-overlapping items (
        mask = ~df1.index.isin(df2.index)
        ball_pos = pd.concat([df1[mask], df2], sort=True)
        # Report results
        print(f'Loaded {len(df1)} stats in {fname_df}')
        print(f'Found {len(df2)} CSV fragments in {pathname}.')
        print(f'Added {np.sum(mask)} new items.')
        
    except:
        ball_pos = ball_pos_new
        # Report results
        print(f'Created new ball_pos.csv with {len(ball_pos)} entries.')
    
    # Save the dataframe
    ball_pos.to_csv('../calculations/ball_pos.csv')


if __name__ == '__main__':
    main()
