import os
import warnings
warnings.simplefilter(action='ignore')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import multiprocessing as mp
import matplotlib.pyplot as plt

def get_blurred_mask(cellmask):
    cellmask = distance_transform_edt(cellmask)
    cellmask = cellmask/np.max(cellmask,axis=None)
    return cellmask

def get_cropped_stack(ims,minR,maxR,minC,maxC,mR,mC):
    imstack = np.zeros([maxR-minR,maxC-minC,len(ims)])
    print(np.shape(imstack))
    cats=[]
    for ii,im in enumerate(ims):
        cat = im.split('/')[-1].split('_')[2]
        cats.append(cat)
        img = imread(os.path.join(im))
        img = img[mR+minR:mR+maxR,mC+minC:mC+maxC]
        imstack[:,:,ii]=img
    return imstack,cats

def get_cell_class(batch):
    predDir = batch['CellSegDir']
    imDir = batch['SAMDir']
    sample = batch['Sample']
    area = batch['Area']
    saveDir = batch['saveDir']
    batchnum = batch['BatchNum']
    cellids = batch['CellIDs']
    cat_dict_nuc = batch['NucCatDict']
    cat_dict_mem = batch['MemCatDict']
    segName = batch['SegName']
    section = segName.split('_')[-1].split('.')[0]
    
    if len(cellids)<1:
        return
    #if os.path.exists(os.path.join(saveDir,sample,area,'_'.join([sample,area,section,'SAM_scores.csv']))):
    #    return
    newname = '_'.join([sample,area,section,'SAM-scores',str(batchnum)+'.csv'])
    if os.path.exists(os.path.join(saveDir,sample,area,newname)):
        return
    if not os.path.exists(os.path.join(imDir,sample,area)):
        print('No class maps for',sample,area,'. Skipping...')
        return
    
    print('Opening cell segs')
    cellsegs = np.load(os.path.join(predDir,sample,area,segName),allow_pickle=True).item()
    print('Opening nuc segs')
    nucsegs = np.load(os.path.join(predDir.replace('wholeCell','nucleus'),sample,area,segName),allow_pickle=True).item()
    
    if 'bbox' in nucsegs.keys():
        mR,_,mC,_=nucsegs['bbox']
    else:
        mR=0
        mC=0
    r,c = np.shape(cellsegs['masks'])
    cellmask = np.where((cellsegs['masks']>=np.min(cellids)) & (cellsegs['masks']<=np.max(cellids)),1,0)
    nucmask = np.where((nucsegs['masks']>=np.min(cellids)) & (nucsegs['masks']<=np.max(cellids)),1,0)
    cellsegs = cellsegs['masks']*cellmask
    nucsegs = nucsegs['masks']*nucmask
    memsegs = cellsegs-nucsegs
    del nucmask
    cellidx = np.where(cellmask==1)
    del cellmask
    minR = np.min(cellidx[0])
    minC = np.min(cellidx[1])
    maxR = np.max(cellidx[0])
    maxC = np.max(cellidx[1])
    R = maxR-minR
    C = maxC-minC
    
    cellsegs = cellsegs[minR:maxR,minC:maxC]
    memsegs = memsegs[minR:maxR,minC:maxC]
    nucsegs = nucsegs[minR:maxR,minC:maxC]
    memcats = list(cat_dict_mem.values())
    nuccats = list(cat_dict_nuc.values())
    
    ims = os.listdir(os.path.join(imDir,sample,area))
    mem_ims = [x for x in ims if x.split('_')[2] in memcats]
    nuc_ims = [x for x in ims if x.split('_')[2] in nuccats]
    mem_ims = [os.path.join(imDir,sample,area,x) for x in mem_ims]
    nuc_ims = [os.path.join(imDir,sample,area,x) for x in nuc_ims]

    tmp = pd.DataFrame()
    tmp['Sample']=[sample]*len(cellids)
    tmp['Area']=[area]*len(cellids)
    tmp['CompName']=[sample+'_'+area]*len(cellids)
    tmp['Section']=[section]*len(cellids)
    tmp['CellID']=cellids

    print('Stacking image channels')
    memstack,_ = get_cropped_stack(mem_ims,minR,maxR,minC,maxC,mR,mC)
    nucstack,_ = get_cropped_stack(nuc_ims,minR,maxR,minC,maxC,mR,mC)
    mem_scores = np.zeros([len(cellids),len(memcats)])
    nuc_scores = np.zeros([len(cellids),len(nuccats)])
    centroidRow=[]
    centroidCol=[]
    print('>>>',len(cellids),'cells')
    #print('Calculating features...')
    for ii,cellid in tqdm(enumerate(cellids)):
        #get cell points and bounding box
        test = np.where(cellsegs==cellid,1,0)
        if np.sum(test,axis=None)==0:
            centroidRow.append(0)
            centroidCol.append(0)
        cell_loc = np.where(cellsegs==cellid)
        rmin = np.min(cell_loc[0])
        rmax = np.max(cell_loc[0])
        cmin = np.min(cell_loc[1])
        cmax = np.max(cell_loc[1])
        centroid = (np.mean(cell_loc[0])+minR+mR,np.mean(cell_loc[1])+minC+mC)
        
        if ((rmax-rmin)<1) or ((cmax-cmin)<1):
            centroidRow.append(centroid[0])
            centroidCol.append(centroid[1])
            continue

        #get single cell crops
        memSAMpatch = memstack[rmin:rmax,cmin:cmax,:]
        nucSAMpatch = nucstack[rmin:rmax,cmin:cmax,:]
        mempatch = memsegs[rmin:rmax,cmin:cmax]
        nucpatch = nucsegs[rmin:rmax,cmin:cmax]

        #get masks and blurred masks
        mempatch = np.where(mempatch==cellid,1,0)
        mempatchw = get_blurred_mask(mempatch)
        nucpatch = np.where(nucpatch==cellid,1,0)
        nucpatchw = get_blurred_mask(nucpatch)

        #get values and weights
        mempatch=mempatch.astype(bool)
        nucpatch=nucpatch.astype(bool)
        memvals = memSAMpatch[mempatch]
        nucvals = nucSAMpatch[nucpatch]
        memw=mempatchw[mempatch].astype(float)
        nucw=nucpatchw[nucpatch].astype(float)

        #mean SAM score over corresponding region
        if np.sum(memw)>0:
            mem_scores[ii, :] = np.average(memvals, axis=0, weights=memw)
        if np.sum(nucw)>0:
            nuc_scores[ii, :] = np.average(nucvals, axis=0, weights=nucw)
        centroidRow.append(centroid[0])
        centroidCol.append(centroid[1])
    
    tmp['CellCentroidRow']=centroidRow
    tmp['CellCentroidCol']=centroidCol
    column_names = memcats+nuccats
    data = np.concatenate([mem_scores,nuc_scores],axis=1)
    tmp[column_names]=data
    tmp.to_csv(os.path.join(saveDir,sample,area,newname))
    print(newname,'SAVED')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='BC_CODEX',help='')
    parser.add_argument('--pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--im_dir',type=str,default='spectral_angle_mapping/class_maps',help='')
    parser.add_argument('--csv_dir',type=str,default='classify_by_dilation',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    parser.add_argument('--cpu_cap',type=int,default=25,help='')

    args,unparsed = parser.parse_known_args()

    PredDir = os.path.join(args.rootdir,args.dataset,args.pred_dir)
    csvDir = os.path.join(args.rootdir,args.dataset,args.csv_dir)
    imDir = os.path.join(args.rootdir,args.dataset,args.im_dir)
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)
    
    reference_spectra = pd.read_csv(os.path.join(args.rootdir,args.dataset,'reference_spectra.csv'))
    cols = [x for x in reference_spectra.columns if 'Unnamed' not in x]
    cols = [x for x in cols if 'arker' not in x]
    cat_dict_mem = {}
    cat_dict_nuc = {}
    for i,col in enumerate(cols):
        if 'Nuc' in col:
            cat_dict_nuc.update({i:col})
        else:
            cat_dict_mem.update({i:col})

    samples = os.listdir(PredDir)
    for sample in samples:
        if not args.sample=='all_samples':
            if args.sample!=sample:
                continue
        if not os.path.exists(os.path.join(csvDir,sample)):
            os.makedirs(os.path.join(csvDir,sample))
        areas = os.listdir(os.path.join(PredDir,sample))
        for area in areas:
            if not args.area=='all_areas':
                if args.area!=area:
                    continue
            if not os.path.exists(os.path.join(csvDir,sample,area)):
                os.makedirs(os.path.join(csvDir,sample,area))
            master_name = sample+'_'+area+'_SAM-scores.csv'
            #if os.path.exists(os.path.join(csvDir,sample,area,master_name)):
            #    continue
            segs = os.listdir(os.path.join(PredDir,sample,area))
            if len(segs)==0:
                continue
            for seg in segs:
                section = seg.split('_')[-1].split('.')[0]
                bnums = []
                for i in range(args.cpu_cap):
                    newname = '_'.join([sample,area,section,'SAM-scores',str(i)+'.csv'])
                    checkPath = os.path.join(csvDir,sample,area,newname)
                    if not os.path.exists(checkPath):
                        bnums.append(i)
                if len(bnums)==0:
                    continue
                print(sample,area,section)
                cellpath = os.path.join(PredDir,sample,area,seg)
                try:
                    cellsegs = np.load(cellpath,allow_pickle=True).item()
                except:
                    cellsegs = np.load(cellpath,allow_pickle=True)
                cells = np.unique(cellsegs['masks'])
                del cellsegs
                cells = np.delete(cells,0)
                batchsize = int(np.ceil(len(cells)/args.cpu_cap))
                cellbatches=[]
                for i in bnums:
                    cellBatch = cells[i*batchsize:(i+1)*batchsize]
                    batchdict = {'CellSegDir':PredDir,'SAMDir':imDir,'Sample':sample,
                                 'saveDir':csvDir,'CellIDs':cellBatch,'Area':area,
                                 'SegName':seg,'BatchNum':i,'NucCatDict':cat_dict_nuc,'MemCatDict':cat_dict_mem}
                    cellbatches.append(batchdict)
                
                if len(cellbatches)>0:
                    num_workers = len(cellbatches)   
                    pool = mp.Pool(num_workers)
                    pool.map(get_cell_class,cellbatches)
                    pool.close()
    
            csvs = os.listdir(os.path.join(csvDir,sample,area))
            csvs = [x for x in csvs if 'SAM-scores_' in x]
            master_name = sample+'_'+area+'_SAM-scores.csv'
            master_df = pd.DataFrame()
            csvs = [x for x in csvs if x!=master_name]
            for csv in csvs:
                df = pd.read_csv(os.path.join(csvDir,sample,area,csv))
                master_df = pd.concat([master_df,df],ignore_index=True)
            keepcols = [x for x in master_df.columns if 'Unnamed' not in x]
            master_df = master_df[keepcols]
            master_df.to_csv(os.path.join(csvDir,sample,area,master_name))
            print(master_name,'saved')
            [os.remove(os.path.join(csvDir,sample,area,x)) for x in csvs if x!=master_name]

    if (args.sample=='all_samples') and (args.area=='all_areas'):
        combined_master_df = pd.DataFrame()
        for sample in samples:
            areas = os.listdir(os.path.join(csvDir,sample))
            for area in areas:
                csvs = os.listdir(os.path.join(csvDir,sample,area))
                csvs = [x for x in csvs if x.endswith('SAM-scores.csv')]
                if len(csvs)>1:
                    print('Too many cell csvs in',sample,area,'. Check files.')
                    continue
                elif len(csvs)==0:
                    print('Skipping sample',sample,area)
                    continue
                df = pd.read_csv(os.path.join(csvDir,sample,area,csvs[0]))
                combined_master_df = pd.concat([combined_master_df,df],ignore_index=True)
        combined_master_df.to_csv(os.path.join(csvDir,'combined_SAM-scores.csv'))


if __name__=='__main__':
    main()
