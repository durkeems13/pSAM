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

def filter_cells(df,mth,nth):
    cats = [x for x in df.columns if 'Int-mean' not in x]
    cats = [x for x in cats if 'Centroid' not in x]
    cats = [x for x in cats if x not in ['Sample','Area','CellID','CompName','Section']]
    cats = [x for x in cats if 'Unnamed' not in x]
    cats = [x for x in cats if 'Cat' not in x]
    memcats = [x for x in cats if 'Nuc' not in x]
    nuccats = [x for x in cats if 'Nuc' in x]
    memmax = list(df[memcats].max(axis=1))
    nucmax = list(df[nuccats].max(axis=1))
    memcatname = list(df[memcats].idxmax(axis=1))
    nuccatname = list(df[nuccats].idxmax(axis=1))
    memcat = [memcats.index(x) for x in memcatname]
    nuccat = [nuccats.index(x) for x in nuccatname]
    memcatname = [x if memmax[i]>mth else 'unclassified' for i,x in enumerate(memcatname)] #0
    nuccatname = [x if nucmax[i]>nth else 'Nucleus' for i,x in enumerate(nuccatname)] #0
    nuccatname = ['Nucleus' if x=='Generic-Nucleus' else x for x in nuccatname]
    finalcatname = [x+'_'+y if y!='Nucleus' else x for x,y in zip(memcatname,nuccatname)]
    finalcatscore = [np.mean([x,y]) for x,y in zip(memmax,nucmax)]

    df['MemCat']=memcat
    df['NucCat']=nuccat
    df['MemCatName']=memcatname
    df['NucCatName']=nuccatname
    df['MemCatScore']=memmax
    df['NucCatScore']=nucmax
    df['FinalCatName']=finalcatname
    df['FinalCatScore']=finalcatscore

    return df    

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
    parser.add_argument('--memth',type=float,default=5,help='')
    parser.add_argument('--nucth',type=float,default=5,help='')
    
    args,unparsed = parser.parse_known_args()
    
    rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
    
    csvname = 'combined_SAM-scores.csv'
    savename = csvname.replace('.csv','_FinalCats.csv')
    
    df = pd.read_csv(os.path.join(rdir,csvname),low_memory=False)
    df = df[[x for x in df.columns if ('Int-mean' not in x) and ('Unnamed' not in x)]]
    #print('Total NaN in dataset',df.isna().sum())
    #[print(x,df[x].isna().sum()) for x in df.columns]
    df = df.fillna(0)
    df = filter_cells(df,args.memth,args.nucth)
    #edit cellIDs for full sections
    sections = df['Section']
    cellids = df['CellID']
    cnames = df['CompName']
    df = fix_split_sections(df)
    df.to_csv(os.path.join(rdir,savename))

    print('MemCats',df.MemCatName.value_counts())
    print('NucCats',df.NucCatName.value_counts())
    pd.set_option('display.max_rows', None)
    print('FinalCats',df.FinalCatName.value_counts())
    pd.reset_option('display.max_rows')
    

if __name__=="__main__":
    main()
