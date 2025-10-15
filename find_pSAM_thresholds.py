#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:03:41 2025

@author: durkeems
"""
import os
import argparse
import numpy as np
from tifffile import imread
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import multiprocessing as mp

def get_threshold(batch):
    timpath = batch['timPath']
    smappaths = batch['smapPaths']
    savepath = batch['savePath']
    binstep = batch['BinStep']

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(os.path.join(savepath,'hists')):
        os.makedirs(os.path.join(savepath,'hists'))

    timg = imread(timpath)
    for smap in smappaths:
        sname = smap.split('/')
        sample = sname[-3]
        area = sname[-2]
        if os.path.exists(os.path.join(savepath,sname[-1].split('_')[-2]+'_hist_th.npy')):
            continue
        img = imread(smap)
        print(np.shape(img))
        r,c = np.shape(img)
        R,C = np.shape(timg)
        if (r!=R) or (c!=C):
            from skimage.transform import resize
            timg = resize(timg,[r,c],preserve_range=True)
        data = img[timg>0]
        del img
        bins = np.arange(0,255,binstep)
        binned_data = [np.count_nonzero(data[data<(x+binstep)]) for x in bins]
        binned_data = [binned_data[i]-binned_data[i-1] for i in np.arange(1,len(binned_data))]
        th,_ = find_peaks(binned_data)
        th = [x for x in th if ((x+2.5)*binstep)>56] #x + 2.5 -> get to top of bin, not 'bottom'
        if len(th)==0:
            th = 127
            d = '--'
        else:
            d = '-'
            thvals = [binned_data[x] for x in th]
            th_ind = np.argmax(thvals)
            th = (th[th_ind]+2.5)*binstep #get to top of bin
            
        thdict = {'sample':sample,'area':area,'dataset':sname[-6],'SAM':sname[-1],'th':th}
        np.save(os.path.join(savepath,sname[-1].split('_')[-2]+'_hist_th.npy'),thdict)
        
        plt.hist(data,bins=len(binned_data))
        plt.axvline(th,color='red',linestyle=d)
        plt.title(smap)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath,'hists',sname[-1].split('_')[-2]+'_hist_th.png'),dpi=200)
        plt.clf()
        print(sname[-1],'DONE')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets')
    parser.add_argument('--dataset',type=str,default='BC_CODEX')
    parser.add_argument('--read',type=str,default='spectral_angle_mapping/class_maps')
    parser.add_argument('--save',type=str,default='spectral_angle_mapping/map_thresholding')
    parser.add_argument('--sample',type=str,default='all_samples')
    parser.add_argument('--area',type=str,default='all_areas')
    parser.add_argument('--tissue',type=str,default='tissue_composite_masks_instance')

    args,_ = parser.parse_known_args()

    rdir = os.path.join(args.rootdir,args.dataset,args.read)
    sdir = os.path.join(args.rootdir,args.dataset,args.save)
    tdir = os.path.join(args.rootdir,args.dataset,args.tissue)
    binstep=10

    if not os.path.exists(sdir):
        os.makedirs(sdir)
    samples = os.listdir(rdir)
    samples.sort()
    samples = [x for x in samples if not x.endswith('.npy')]
    tims = os.listdir(tdir)
    batches = []
    for sample in samples:
        if (args.sample!='all_samples') and (args.sample!=sample):
            continue
        if not os.path.exists(os.path.join(sdir,sample)):
            os.makedirs(os.path.join(sdir,sample))
        areas = os.listdir(os.path.join(rdir,sample))
        areas.sort()
        for area in areas:
            if (args.area!="all_areas") and (args.area!=area):
                continue
            if not os.path.exists(os.path.join(sdir,sample,area)):
                os.makedirs(os.path.join(sdir,sample,area))
            if not os.path.exists(os.path.join(sdir,sample,area,'hists')):
                os.makedirs(os.path.join(sdir,sample,area,'hists'))
            smaps = os.listdir(os.path.join(rdir,sample,area))
            if len(os.listdir(os.path.join(sdir,sample,area,'hists')))==len(smaps):
                continue
            tim = [x for x in tims if (sample in x) and (area in x)]
            if len(tim)==0:
                print('No tissue segs for',sample,area,'. Continuing...')
                continue
            timPath = os.path.join(tdir,tim[0])
            smapPaths = [os.path.join(rdir,sample,area,x) for x in smaps]
            savePath = os.path.join(sdir,sample,area)
            batches.append({'timPath':timPath,'smapPaths':smapPaths,
                            'savePath':savePath,'BinStep':binstep})
    pool = mp.Pool(len(batches))
    pool.map(get_threshold,batches)
    pool.close()
                
if __name__=='__main__':
    main()
