#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:49:49 2023

@author: durkeems
"""
import os
import argparse
import re
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu,threshold_multiotsu
import matplotlib.pyplot as plt
from typing import Iterable, List, Dict, Optional, Tuple

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

def plot_mpi_by_class(
    df: pd.DataFrame,
    sdir: str,
    memname_dict: dict | None = None,
    nucname_dict: dict | None = None,
):
    """
    Builds mean-intensity and z-score heatmaps per predicted class.

    Parameters
    ----------
    df : DataFrame
        Must include columns:
          - 'MemCatName', 'NucCatName'
          - intensity columns like '*Int-mean_wc*'
    sdir : str
        Output directory for PNGs.
    cols : list[str]
        Candidate column names to filter from.
    memname_dict : dict
        Optional pretty-name mapping for membrane classes.
    nucname_dict : dict
        Optional pretty-name mapping for nucleus classes.
    """

    os.makedirs(sdir, exist_ok=True)

    # ---- categories (preserve np.unique ordering) ----
    memcats = np.unique(df['MemCatName'].to_numpy())
    nuccats = np.unique(df['NucCatName'].to_numpy())

    # ---- select marker-intensity columns ----
    mpicols = [c for c in df.columns if ('Int-mean_wcw' in c) and ('TAF' not in c) and ('DIC' not in c)]
    mpicols = [c for c in mpicols if 'BCDA2' not in c]
    mpicols.sort()
    mpidf = df[mpicols]

    # ---- z-score (column-wise) without modifying df ----
    col_means = mpidf.mean(axis=0)
    col_stds = mpidf.std(axis=0).replace(0, np.nan)  # avoid divide-by-zero -> NaN z (will average to NaN)
    zdf = (mpidf - col_means) / col_stds

    # ---- group means by class (vectorized) ----
    # means of raw intensities
    mu_mem = df.groupby('MemCatName', sort=False)[mpicols].mean().reindex(memcats)
    mu_nuc = df.groupby('NucCatName', sort=False)[mpicols].mean().reindex(nuccats)
    mu_array = np.vstack([mu_mem.to_numpy(), mu_nuc.to_numpy()])

    # means of z-scores (align with df index first)
    zdf_aligned = zdf.set_index(df.index)
    z_mem = zdf_aligned.groupby(df['MemCatName'], sort=False).mean().reindex(memcats)
    z_nuc = zdf_aligned.groupby(df['NucCatName'], sort=False).mean().reindex(nuccats)
    z_array = np.vstack([z_mem.to_numpy(), z_nuc.to_numpy()])

    # ---- labels ----
    # friendly class names if dicts provided; fall back to originals
    memname_dict = memname_dict or {}
    nucname_dict = nucname_dict or {}
    # maintain your special-case tweak
    nucname_dict = {**nucname_dict, 'GenericNucleus': 'Generic-Nucleus'}

    cats = [memname_dict.get(x, x) for x in memcats] + [nucname_dict.get(x, x) for x in nuccats]

    # marker names
    markers = [c.split('_Int')[0] for c in mpicols]
    markers = [m.replace('gamma', '\u03B3').replace('delta', '\u03B4') for m in markers]

    # ---- plot 1: means ----
    plt.figure(figsize=(4, 4), dpi=600)
    plt.imshow(mu_array, cmap='jet', aspect='auto')
    plt.yticks(ticks=np.arange(len(cats)), labels=cats, fontsize=5)
    plt.xticks(ticks=np.arange(len(mpicols)), labels=markers, rotation=90, fontsize=5)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel('Predicted Cell Class', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, 'MeanMPI-perClass.png'), dpi=600)
    plt.close()

    # ---- plot 2: z-scores (clamped -1..1 like before) ----
    plt.figure(figsize=(6.5, 4.5), dpi=600)
    plt.imshow(z_array, vmin=-1, vmax=1, cmap='seismic', aspect='auto')
    plt.yticks(ticks=np.arange(len(cats)), labels=cats, fontsize=8)
    plt.xticks(ticks=np.arange(len(mpicols)), labels=markers, rotation=90, fontsize=8)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=11)
    plt.ylabel('Predicted Cell or Nucleus Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, 'MeanMPIzscore-perClass.png'), dpi=600)
    plt.close()

def _ordered_unique(seq: Iterable[str]) -> List[str]:
    """Stable unique preserving first occurrence."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def anonymize_compnames(
    comp_series: pd.Series,
    sep: str = "_",
    area_from: str = "Area",
    area_to: str = "A",
) -> Tuple[pd.Series, Dict[str, str]]:
    """
    Anonymize CompName by mapping the *prefix* (before sep) to S1, S2, ...
    and replacing 'Area' with 'A' in the final strings (to match your behavior).
    Returns the anonymized series and the mapping dict {orig_prefix: 'S#'}.
    """
    comp_series = comp_series.astype(str)
    prefixes = comp_series.str.split(sep, n=1, expand=True)[0]
    uniq_prefixes = _ordered_unique(prefixes.tolist())
    prefix_map = {p: f"S{i+1}" for i, p in enumerate(uniq_prefixes)}

    # Replace only the prefix occurrence at start-of-string
    anon = comp_series.copy()
    for orig, anonp in prefix_map.items():
        anon = anon.str.replace(rf"^{re.escape(orig)}", anonp, regex=True)

    # Area -> A, like your original
    anon = anon.str.replace(area_from, area_to, regex=False)
    return anon, prefix_map


def anonymize_ids(series: pd.Series, label_prefix: str = "S") -> Tuple[pd.Series, Dict[str, str]]:
    """
    One-to-one anonymization of IDs (e.g., AccNum) to S1..SN in first-seen order.
    Returns the anonymized series and {orig: 'S#'} mapping.
    """
    series = series.astype(str)
    uniques = _ordered_unique(series.tolist())
    id_map = {u: f"{label_prefix}{i+1}" for i, u in enumerate(uniques)}
    return series.map(id_map), id_map


def compute_category_proportions(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    ordered_cats: Iterable[str],
) -> pd.DataFrame:
    """
    Vectorized per-group proportions of category_col. Missing cats -> 0.
    Output columns are intersect(ordered_cats, observed) in that order.
    """
    vc = (
        df.groupby(group_col)[category_col]
          .value_counts(normalize=True)
          .unstack(fill_value=0.0)
    )
    # Keep only requested cats that appear; ensure stable requested order
    keep_cols = [c for c in ordered_cats if c in vc.columns]
    return vc.reindex(columns=keep_cols)


def resolve_colors_in_order(
    color_dict: Dict[str, Iterable[float]],
    ordered_cats: Iterable[str],
) -> List[str]:
    """Return a hex color list parallel to ordered_cats (skipping cats not found)."""
    cols = []
    for cat in ordered_cats:
        if cat in color_dict:
            cols.append(rgb_to_hex(color_dict[cat]))
    return cols


def plot_stacked_bar(
    prop_df: pd.DataFrame,
    xlabels: Optional[List[str]],
    outfile: str,
    title: Optional[str] = None,
    legend: bool = False,
    xlabel: str = "",
    ylabel: str = "Proportion of Cells",
    tick_fontsize: int = 11,
    ylabel_fontsize: int = 12,
    xlabel_fontsize: int = 12,
    width: float = 0.9,
    dpi: int = 600,
):
    """
    Plot a stacked bar from a proportions DataFrame (index = groups, columns = categories).
    """
    plt.figure(figsize=(6.5, 4.5), dpi=dpi)
    # Pandas plot wants colors via prop_df.plot(..., color=[...])
    ax = prop_df.plot(
        kind="bar",
        stacked=True,
        width=width,
        color=None,   # set through ax containers later (keeps function generic)
        legend=legend,
        ax=plt.gca(),
        rot=0,
    )

    # Optional relabeling of x-axis (e.g., anonymized labels)
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=90, fontsize=tick_fontsize)
    else:
        ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=90)

    # Y axis formatting
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Labels & title
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()


def set_bar_colors(ax: plt.Axes, hex_colors: List[str]):
    """
    Assign a color per category stack using the plotting order of bar containers.
    Assumes len(hex_colors) == number of category columns in the plotted DataFrame.
    """
    containers = [c for c in ax.containers if hasattr(c, "datavalues")]
    for cont, color in zip(containers, hex_colors):
        for patch in cont.patches:
            patch.set_facecolor(color)


def load_nucleus_color_dict(npy_path: str) -> Dict[str, Iterable[float]]:
    """
    Load color_dictionary.npy and filter to nucleus entries.
    Rename 'Nucleus' -> 'Generic-Nucleus' (your convention).
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Color dictionary not found: {npy_path}")

    cdict = np.load(npy_path, allow_pickle=True).item()
    nuc_names = [k for k in cdict.keys() if "Nucleus" in k or k == "Nucleus"]
    out = {}
    for name in nuc_names:
        new_name = "Generic-Nucleus" if name == "Nucleus" else name
        out[new_name] = cdict[name]
    return out


def make_prevalence_plots(
    df: pd.DataFrame,
    sdir: str,
    csvname: str,
    memcolor_dict: Dict[str, Iterable[float]],
    mem_cat_col: str = "MemCatName",
    nuc_cat_col: str = "NucCatName",
    comp_col: str = "CompName",
    acc_col: str = "AccNum",
    color_dict_npy: Optional[str] = None,  # e.g., os.path.join(root,dataset,'color_dictionary.npy')
    order_dict_npy: Optional[str] = None,  # e.g., os.path.join(root,dataset,'color_dictionary.npy')
):
    """
    Generates four stacked-bar prevalence plots (membrane & nucleus) at both image and sample levels.
    Adds AnonCompName and AnonAccNum columns to df and returns the modified DataFrame.

    Outputs:
      - CellTypePrevalence_{csvname}.png        (Membrane by anonymized CompName)
      - SampleCellTypePrevalence_{csvname}.png  (Membrane by AccNum)
      - NucleusTypePrevalence_{csvname}.png     (Nucleus by anonymized CompName)
      - SampleNucleusTypePrevalence_{csvname}.png (Nucleus by AccNum)
    """
    os.makedirs(sdir, exist_ok=True)

    # ---------- Anonymize CompName & AccNum ----------
    anon_comp, _ = anonymize_compnames(df[comp_col])
    anon_acc, _ = anonymize_ids(df[acc_col], label_prefix="S")

    df = df.copy()
    df["AnonCompName"] = anon_comp
    df["AnonAccNum"] = anon_acc

    # ---------- Ordered category lists ----------
    ordered_mem_cats = np.load(order_dict_npy,allow_pickle=True)
    if 'unclassified' not in ordered_mem_cats:
        ordered_mem_cats = list(ordered_mem_cats)+['unclassified']
    ordered_mem_cats = [x for x in ordered_mem_cats if 'Nucleus' not in x]
    print(ordered_mem_cats)

    # ---------- Membrane: by AnonCompName ----------
    mem_by_img = compute_category_proportions(
        df.assign(Group=df["AnonCompName"]),
        group_col="Group",
        category_col=mem_cat_col,
        ordered_cats=ordered_mem_cats,
    )
    mem_img_colors = resolve_colors_in_order(memcolor_dict, ordered_mem_cats)
    outfile = os.path.join(sdir, f"CellTypePrevalence_{csvname}.png")

    plt.figure(figsize=(6.5, 4.5), dpi=600)
    ax = mem_by_img.plot(
        kind="bar",
        stacked=True,
        width=0.9,
        legend=False,
        rot=0,
        ax=plt.gca(),
    )
    set_bar_colors(ax, mem_img_colors)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Proportion of Cells", fontsize=12)
    ax.set_xticklabels(mem_by_img.index.tolist(), fontsize=11, rotation=90)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Membrane Type Prevalence")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

    # ---------- Membrane: by AccNum ----------
    mem_by_sample = compute_category_proportions(
        df.assign(Group=df[acc_col].astype(str)),
        group_col="Group",
        category_col=mem_cat_col,
        ordered_cats=ordered_mem_cats,
    )
    mem_sample_colors = resolve_colors_in_order(memcolor_dict, ordered_mem_cats)
    outfile = os.path.join(sdir, f"SampleCellTypePrevalence_{csvname}.png")

    plt.figure(figsize=(6.5, 4.5), dpi=600)
    ax = mem_by_sample.plot(
        kind="bar",
        stacked=True,
        width=0.9,
        legend=False,
        rot=0,
        ax=plt.gca(),
    )
    set_bar_colors(ax, mem_sample_colors)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Proportion of Cells", fontsize=16)
    ax.set_xticklabels(mem_by_sample.index.tolist(), fontsize=14, rotation=90)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

    # ---------- Nucleus color dict ----------
    if color_dict_npy is not None:
        nuccolor_dict = load_nucleus_color_dict(color_dict_npy)
    else:
        # If none provided, try to infer from membrane dict keys containing 'Nucleus' (fallback)
        nuccolor_dict = {k: v for k, v in memcolor_dict.items() if "Nucleus" in k or k == "Nucleus"}
        '''
        if "Nucleus" in nuccolor_dict:
            nuccolor_dict["Generic-Nucleus"] = nuccolor_dict.pop("Nucleus")
        '''
    if not nuccolor_dict:
        raise ValueError("No nucleus color dictionary available. Provide color_dict_npy or include nucleus keys.")

    ordered_nuc_cats = list(nuccolor_dict.keys())
    # ---------- Nucleus: by AnonCompName (use anonymized label on x-axis) ----------
    nuc_by_img = compute_category_proportions(
        df.assign(Group=df["AnonCompName"]),
        group_col="Group",
        category_col=nuc_cat_col,
        ordered_cats=ordered_nuc_cats,
    )
    nuc_img_colors = resolve_colors_in_order(nuccolor_dict, nuc_by_img.columns)
    outfile = os.path.join(sdir, f"NucleusTypePrevalence_{csvname}.png")

    plt.figure(figsize=(6.5, 4.5), dpi=600)
    ax = nuc_by_img.plot(
        kind="bar",
        stacked=True,
        width=0.9,
        legend=False,
        rot=0,
        ax=plt.gca(),
    )
    set_bar_colors(ax, nuc_img_colors)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Proportion of Cells", fontsize=12)
    ax.set_xticklabels(nuc_by_img.index.tolist(), fontsize=11, rotation=90)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Nucleus Class Prevalence")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

    # ---------- Nucleus: by AccNum ----------
    nuc_by_sample = compute_category_proportions(
        df.assign(Group=df[acc_col].astype(str)),
        group_col="Group",
        category_col=nuc_cat_col,
        ordered_cats=ordered_nuc_cats,
    )
    nuc_sample_colors = resolve_colors_in_order(nuccolor_dict, nuc_by_sample.columns)
    outfile = os.path.join(sdir, f"SampleNucleusTypePrevalence_{csvname}.png")

    plt.figure(figsize=(6.5, 4.5), dpi=600)
    ax = nuc_by_sample.plot(
        kind="bar",
        stacked=True,
        width=0.9,
        legend=False,
        rot=0,
        ax=plt.gca(),
    )
    set_bar_colors(ax, nuc_sample_colors)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Proportion of Cells", fontsize=16)
    ax.set_xticklabels(nuc_by_sample.index.tolist(), fontsize=14, rotation=90)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

    
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
    sdir = os.path.join(rdir,'Prevalence_and_Validation_Plots')
    
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    csv_name = 'combined_SAM-scores.csv'
    csvname = csv_name.replace('.csv','_FinalCats.csv')
    
    if not os.path.exists(os.path.join(rdir,'_'.join([args.dataset,'cellMPI','FinalCats.csv']))):
        if not os.path.exists(os.path.join(rdir,csvname)):
            SAMdf = pd.read_csv(os.path.join(rdir,csv_name),low_memory=False)
            SAMdf = SAMdf.fillna(0)
            SAMdf = filter_cells(SAMdf,args.memth,args.nucth)
            SAMdf.to_csv(os.path.join(rdir,csvname))
        else:
            SAMdf = pd.read_csv(os.path.join(rdir,csvname))
        mpi_df = pd.read_csv(os.path.join(rdir,'combined_cellMPI.csv'))
        mpi_df = fix_split_sections(mpi_df)
        df = SAMdf.merge(mpi_df,on=['FullSection','AdjCellID'])
        keepcols = [x for x in df.columns if not x.endswith('_y')]
        renamecols = [x for x in df.columns if x.endswith('_x')]
        newnames = [x.replace('_x','') for x in renamecols]
        df=df[keepcols]
        df[newnames]=df[renamecols]
        df=df.drop(columns=renamecols)
        print('SAM df:',SAMdf.shape)
        print('MPI df:',mpi_df.shape)
        print('Combined df:',df.shape)
        df.to_csv(os.path.join(rdir,'_'.join([args.dataset,'cellMPI','FinalCats.csv'])))
    else:
        df = pd.read_csv(os.path.join(rdir,'_'.join([args.dataset,'cellMPI','FinalCats.csv'])))
    
    if 'AccNum' not in df.columns:
        df['AccNum']=df['Sample']

    color_npy = os.path.join(args.rootdir, args.dataset, "color_dictionary.npy")
    order_npy = os.path.join(args.rootdir, args.dataset, "cell_class_order.npy")
    if os.path.exists(color_npy):
        colordict = np.load(color_npy,allow_pickle=True).item()
        kys = list(colordict.keys())
        memkys = [x for x in kys if 'Nucleus' not in x]
        memcolor_dict = {x:colordict[x] for x in memkys}
        if 'unclassified' not in memcolor_dict.keys():
            memcolor_dict.update({'unclassified':[128,128,128]})
            memkys.append('unclassified')
        
        nuckys = [x for x in kys if 'Nucleus' in x]
        nuccolor_dict = {x:colordict[x] for x in nuckys}
        nuckys = ['Generic-Nucleus' if x=='Nucleus' else x for x in nuckys]
        memname_dict = {x:x for x in memkys}
        nucname_dict = {ky:ky for ky in nuckys}
    else:
        print('Please create a color dictionary and save it as a .npy file. Exiting script.')
 
    make_prevalence_plots(
        df=df,
        sdir=sdir,
        csvname=csvname,
        memcolor_dict=memcolor_dict,
        color_dict_npy=color_npy,   # or None
        order_dict_npy=order_npy,   # or None
    )
    
    plot_mpi_by_class(
        df=df,
        sdir=sdir,
        memname_dict=memname_dict,    # or {}
        nucname_dict=nucname_dict,    # or {}
    )

      
    
if __name__=="__main__":
    main()
