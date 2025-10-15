#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:43:55 2025

@author: durkeems
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np   

def plot_scatter_by_group(df, color_dict_path, output_dir="plots"):
    """
    Create scatter plots for each unique (CompName, Section).
    
    Parameters:
    df : pandas.DataFrame
        Must contain columns: CompName, Section, CellCentroidRow, CellCentroidCol, MemCatName
    output_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get unique groups
    #groups = df.groupby(["CompName", "Section"])
    groups = df.groupby("CompName")
    
    # Load the color dictionary
    color_dict = np.load(color_dict_path, allow_pickle=True).item()
    color_dict['unclassified']=[127,127,127]

    # Convert RGB (0–255 or 0–1) to matplotlib-usable format
    palette = {}
    for k, rgb in color_dict.items():
        rgb = np.array(rgb, dtype=float)
        if rgb.max() > 1.0:  # assume 0–255 scale
            rgb = rgb / 255.0
        palette[k] = tuple(rgb)

    #for (comp, section), group in groups:
    for (comp), group in groups:
        plt.figure(figsize=(6,6))  # square figure
        ax = sns.scatterplot(
            data=group,
            x="CellCentroidCol",
            y="CellCentroidRow",
            hue="MemCatName",
            palette=palette,
            alpha=0.5,
            s=8,
            linewidth=0
        )
        ax.set_aspect('equal', adjustable='box')
        #plt.title(f"{comp} - {section}")
        plt.title(f"{comp}")

        # Get legend and resize markers
        # Get legend and resize markers
        leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="MemCatName")
        if leg is not None:
            for handle in leg.legendHandles:
                handle.set_markersize(8)  # increase marker size in legend
        plt.tight_layout()
        # Save file
        fname = f"{comp}.png".replace(" ", "_")
        
        #fname = f"{comp} - {section}.png".replace(" ", "_")
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()

rdir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/BC_CODEX/classify_by_dilation'
sdir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/BC_CODEX/classification_visual_check'
cdict_path = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/BC_CODEX/color_dictionary.npy'
flist = os.listdir(rdir)
flist = [x for x in flist if x.endswith('FinalCats.csv')]
df = pd.read_csv(os.path.join(rdir,flist[0]))

plot_scatter_by_group(df, cdict_path, output_dir=sdir)
