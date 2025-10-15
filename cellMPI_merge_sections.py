#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:49:49 2023

@author: durkeems
"""
import os
import argparse
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu,threshold_multiotsu
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
  return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def fix_split_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: CompName, Section, CellID
    Returns a copy with new columns: FullSection, AdjCellID
    """
    out = df.copy()

    # Base section name with any single trailing 'a' or 'b' removed (seg1a -> seg1; seg2 -> seg2)
    out['BaseSection'] = out['Section'].astype(str).str.replace(r'[ab]$', '', regex=True)

    # FullSection = CompName + '_' + BaseSection
    out['FullSection'] = out['CompName'].astype(str) + '_' + out['BaseSection']

    # Identify split halves
    is_a = out['Section'].astype(str).str.endswith('a')
    is_b = out['Section'].astype(str).str.endswith('b')

    # For each (CompName, BaseSection), find the max CellID in the 'a' half, then add 1 for a safe offset
    key = ['CompName', 'BaseSection']
    offset_series = (
        out.loc[is_a]
        .groupby(key)['CellID']
        .max()
        .add(1)  # +1 so AdjCellIDs are unique across a/b without colliding at the boundary
        .rename('offset')
    )

    # Map offsets onto all rows (NaN -> 0 for non-split or missing 'a')
    out = out.merge(offset_series, how='left', left_on=key, right_index=True)
    out['offset'] = out['offset'].fillna(0).astype(int)

    # Adjust only the 'b' rows; 'a' and unsplit sections remain unchanged
    out['AdjCellID'] = out['CellID'] + np.where(is_b, out['offset'], 0)
    out['AdjCellID'] = out['AdjCellID'].astype(int)

    # Clean up helpers if you like
    out.drop(columns=['BaseSection', 'offset'], inplace=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/',help='')
    parser.add_argument('--dataset',type=str,default='BC_CODEX',help='')
    parser.add_argument('--readdir',type=str,default='classify_by_dilation',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    
    args,unparsed = parser.parse_known_args()
    
    rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
    
    csvname = 'combined_cellMPI.csv'
    
    df = pd.read_csv(os.path.join(rdir,csvname),low_memory=False)
    df = df[[x for x in df.columns if 'Unnamed' not in x]]
    #edit cellIDs for full sections
    sections = df['Section']
    cellids = df['CellID']
    cnames = df['CompName']
    df = fix_split_sections(df)
    df.to_csv(os.path.join(rdir,csvname))


if __name__=="__main__":
    main()
