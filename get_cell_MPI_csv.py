import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
import multiprocessing as mp

def calculate_cell_feats(batches):
    for batch in batches:
        predDir = batch['predDir']
        imDir = batch['ImDir']
        sample = batch['Sample']
        area = batch['Area']
        saveDir = batch['saveDir']
        batchnum = batch['BatchNum']
        
        dset_dict = {'PSC_CRC':'PSC','Lupus_Nephritis':'LuN',
                     'Renal_Allograft':'MR','TNBC':'TNBC',
                     'Normal_Kidney':'NK'}
        dataset = imDir.split('/')[-2]
        dset = dset_dict[dataset]

        cellids = batch['CellIDs']
        compname = '_'.join([sample,area])
        
        save_df = pd.DataFrame()
        predName = os.listdir(os.path.join(predDir,sample,area))
        cellsegs = np.load(os.path.join(predDir,sample,area,predName[0]),allow_pickle=True).item()
        cellsegs = cellsegs['masks']
        
        r,c = np.shape(cellsegs)
        cellmask = np.where((cellsegs>=np.min(cellids)) & (cellsegs<=np.max(cellids)),1,0)
        
        cellidx = np.where(cellmask==1)
        minR = np.min(cellidx[0])
        minC = np.min(cellidx[1])
        maxR = np.max(cellidx[0])
        maxC = np.max(cellidx[1])
        cellsegs = cellsegs[minR:maxR,minC:maxC]
        
        if len(cellids)==0:
            cellids = np.unique(cellsgs)
            cellids = np.delete(cellids,0)

        ims = os.listdir(os.path.join(imDir,sample,area))
        tmp = pd.DataFrame()
        tmp['Sample']=[sample]*len(cellids)
        tmp['Area']=[area]*len(cellids)
        tmp['CompName']=[sample+'_'+area]*len(cellids)
        tmp['CellID']=cellids
        for ii,im in enumerate(ims):
            stain = im.split('_')[0]
            img = imread(os.path.join(imDir,sample,area,im))
            img = img[minR:maxR,minC:maxC]
            mpi_vec = []
            for jj,cell in tqdm(enumerate(cellids)):
                mpi = np.mean(img[cellsegs==cell])
                mpi_vec.append(mpi)
            tmp[stain+'_Int-mean']=mpi_vec
        save_df = pd.concat([save_df,tmp],ignore_index=True)
        newname = '_'.join([sample,area,'cellMPI',str(batchnum)+'.csv'])
        print(newname)
        save_df.to_csv(os.path.join(saveDir,sample,area,newname))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='Lupus_Nephritis',help='')
    parser.add_argument('--pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--im_dir',type=str,default='Normalized_composites',help='')
    parser.add_argument('--save_dir',type=str,default='with-MPI',help='')
    parser.add_argument('--SAMcsv_dir',type=str,default='classify_by_dilation',help='')
    parser.add_argument('--sample',type=str,default='012523S4',help='')
    parser.add_argument('--area',type=str,default='Area2',help='')
    parser.add_argument('--patch_pad',type=float,default=1,help='in microns')
    parser.add_argument('--pxsize',type=float,default=0.1507,help='in microns')
    parser.add_argument('--cpu_cap',type=int,default=50,help='')

    args,unparsed = parser.parse_known_args()

    PredDir = os.path.join(args.rootdir,args.dataset,args.pred_dir)
    SAMDir = os.path.join(args.rootdir,args.dataset,args.SAMcsv_dir)
    imDir = os.path.join(args.rootdir,args.dataset,args.im_dir)
    saveDir = os.path.join(args.rootdir,args.dataset,args.SAMcsv_dir,args.save_dir)
    
    predNames=[]
    samples = os.listdir(PredDir)
    for sample in samples:
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        areas = os.listdir(os.path.join(PredDir,sample))
        for area in areas:
            if args.area!='all_areas':
                if args.area!=area:
                    continue
            preds = os.listdir(os.path.join(PredDir,sample,area))
            predNames.extend(preds)
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    cellbatches=[]
    for sample in samples:
        if 'S15' in sample:
            continue
        if not args.sample=='all_samples':
            if args.sample!=sample:
                continue
        if not os.path.exists(os.path.join(saveDir,sample)):
            os.makedirs(os.path.join(saveDir,sample))
        areas = os.listdir(os.path.join(PredDir,sample))
        for area in areas:
            if not args.area=='all_areas':
                if args.area!=area:
                    continue
            if not os.path.exists(os.path.join(saveDir,sample,area)):
                os.makedirs(os.path.join(saveDir,sample,area))
            if (args.sample!='all_samples') and (args.area!='all_areas'):
                segs = os.listdir(os.path.join(PredDir,sample,area))
                cellsegs = np.load(os.path.join(PredDir,sample,area,segs[0]),allow_pickle=True).item()
                cells = np.unique(cellsegs['masks'])
                del cellsegs
                cells = np.delete(cells,0)
                batchsize = int(np.ceil(len(cells)/args.cpu_cap))
            else:
                cells = []
                batchsize=1
            for i in range(args.cpu_cap):
                if batchsize>1:
                    cellBatch = cells[i*batchsize:(i+1)*batchsize]
                    batchdict = {'predDir':PredDir,'ImDir':imDir,'Sample':sample,
                                 'saveDir':saveDir,'CellIDs':cellBatch,'Area':area,
                                 'BatchNum':i,'CompName':None}
                else:
                    batchdict = {'predDir':PredDir,'ImDir':imDir,'Sample':sample,
                                 'saveDir':saveDir,'CellIDs':cells,'Area':area,
                                 'BatchNum':i,'CompName':None}
                cellbatches.append(batchdict)
                
    num_workers = args.cpu_cap   
    batchsize = int(np.ceil(len(cellbatches)/num_workers))
    
    batches = []
    for i in range(num_workers):
        Batch = cellbatches[i*batchsize:(i+1)*batchsize]
        batches.append(Batch)

    pool = mp.Pool(num_workers)
    pool.map(calculate_cell_feats,batches)
    pool.close()
    
    if (args.sample!='all_samples') and (args.area!='all_areas'):
        csvs = os.listdir(os.path.join(saveDir,args.sample,args.area))
        csvs = [x for x in csvs if x.endswith('.csv')]
        master_name = args.sample+'_'+args.area+'_cellMPI.csv'
        master_df = pd.DataFrame()
        csvs = [x for x in csvs if x!=master_name]
        for csv in csvs:
            df = pd.read_csv(os.path.join(saveDir,args.sample,args.area,csv))
            master_df = pd.concat([master_df,df],ignore_index=True)
    else:
        return
    keepcols = [x for x in master_df.columns if 'Unnamed' not in x]
    master_df = master_df[keepcols]
    if (args.sample!='all_samples') and (args.area!='all_areas'):
        master_df.to_csv(os.path.join(saveDir,args.sample,args.area,master_name))
        [os.remove(os.path.join(saveDir,args.sample,args.area,x)) for x in csvs if x!=master_name]
    else:
        master_df.to_csv(os.path.join(saveDir,master_name))
        [os.remove(os.path.join(saveDir,x)) for x in csvs if x!=master_name]

if __name__=='__main__':
    main()
