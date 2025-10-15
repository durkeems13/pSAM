import os
import argparse
import numpy as np
import pandas as pd
from tifffile import imread,imwrite
import multiprocessing as mp
from tqdm import tqdm
from skimage.measure import block_reduce
import matplotlib.pyplot as plt


def make_full_colorbar(memnames,nucnames,memcolordict,nuccolordict,memname_dict,nucname_dict,sdir):
    disp_names=[]
    color_dict={}
    for i,mname in enumerate(memnames):
        val = memcolordict[mname]
        color_dict.update({mname:val})
        disp_names.append(memname_dict[mname])
    for i,nname in enumerate(nucnames):
        val = nuccolordict[nname]
        color_dict.update({nname:val}) 
        disp_names.append(nucname_dict[nname])
    CL = np.array(list(color_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_names)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_names,rotation=0,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,'full_colorbar.png'))
    plt.close() 

def make_nucleus_colorbar(nucnames,sdir,dset,colorpath):
    nuccolor_dict={}
    nucname_dict={}
    #selected nucleus colors
    nuccolordict=np.load(os.path.join(colorpath,'color_dictionary.npy'),allow_pickle=True).item()
    nuccolordict['Nucleus']=[128,128,128]
    nuc_names = [x.replace('-',' ') for x in nuccolordict.keys()]
    nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolordict.keys()))}
    disp_nucnames=[]
    for i,nname in enumerate(nucnames):
        val = nuccolordict[nname]
        nuccolor_dict.update({nname:val}) 
        disp_nucnames.append(nucname_dict[nname])
    CL = np.array(list(nuccolor_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,1.5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_nucnames)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_nucnames,rotation=0,fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,'nucleus_colorbar.png'))
    plt.close() 
    
    return nuccolordict,nucname_dict

def make_membrane_colorbar(memnames,sdir,dset,colorpath):
    memcolor_dict = {}
    memname_dict = {}
    memcolordict=np.load(os.path.join(colorpath,'color_dictionary.npy'),allow_pickle=True).item()
    memcolordict['unclassified']=[128,128,128]
    mem_names = [x.replace('-',' ') for x in memcolordict.keys()]
    memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}
    memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}
    disp_memnames=[]
    for i,mname in enumerate(memnames): 
        val = memcolordict[mname]
        memcolor_dict.update({mname:val})
        disp_memnames.append(memname_dict[mname])
    
    CL = np.array(list(memcolor_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_memnames)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_memnames,rotation=0,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,'membrane_colorbar.png'))
    #plt.show()
    plt.close()

    return memcolordict,memname_dict

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def map_to_composite(batch):
    cellPredDir = batch['cellPredDir']
    nucPredDir = batch['nucPredDir']
    sample = batch['Sample']
    area = batch['Area']
    PredName = batch['PredName']
    section = batch['Section']

    mem_maskName = batch['MemMaskName']
    nuc_maskName = batch['NucMaskName']
    memscore_maskName = mem_maskName.replace('.tif','_score.tif')
    nucscore_maskName = nuc_maskName.replace('.tif','_score.tif')

    saveDir = batch['SaveDir']
    catdf = batch['CatDF']
    memcolor_dict = batch['MemColorDict']
    nuccolor_dict = batch['NucColorDict']
    minR,maxR,minC,maxC = batch['Bbox']
    
    if not os.path.exists(os.path.join(cellPredDir,sample,area)):
        print('Whole cell segmentations have not yet been completed for this sample. Displaying only nucleus segmentations...')
        return
    
    mem_pred_comp = np.load(os.path.join(cellPredDir,sample,area,PredName),allow_pickle=True).item()
    mem_pred_comp = mem_pred_comp['masks']
    nuc_pred_comp = np.load(os.path.join(nucPredDir,sample,area,PredName),allow_pickle=True).item()
    nuc_pred_comp = nuc_pred_comp['masks']

    R = maxR-minR
    C = maxC-minC
    
    mem_pred_comp = mem_pred_comp[minR:maxR,minC:maxC]
    nuc_pred_comp = nuc_pred_comp[minR:maxR,minC:maxC]
    
    mem_mask = np.zeros([R,C,3])
    mem_score_mask = np.zeros([R,C,3])
    nuc_mask = np.zeros([R,C,3])
    nuc_score_mask = np.zeros([R,C,3])
    
    cells = list(catdf['CellID'])
    for cell in tqdm(cells):
        cmask = np.where(mem_pred_comp==cell,1,0)
        if np.sum(cmask,axis=None)==0:
            continue
        nmask = np.where(nuc_pred_comp==cell,1,0)
        celldf = catdf[catdf['CellID']==cell]
        if celldf.shape[0]>1:
            celldf = celldf.iloc[0]
            cellcat_mem = celldf['MemCatName']
            cellcat_nuc = celldf['NucCatName']
        else:
            memcellcats = list(celldf['MemCatName'])
            nuccellcats = list(celldf['NucCatName'])
            cellcat_mem = memcellcats[0]
            cellcat_nuc = nuccellcats[0]
            
        cellcolor_mem = memcolor_dict[cellcat_mem]
        cellcolor_nuc = nuccolor_dict[cellcat_nuc]
        
        mem_mask[:,:,0]+=(cmask*cellcolor_mem[0])
        mem_mask[:,:,1]+=(cmask*cellcolor_mem[1])
        mem_mask[:,:,2]+=(cmask*cellcolor_mem[2])
        
        nuc_mask[:,:,0]+=(nmask*cellcolor_nuc[0])
        nuc_mask[:,:,1]+=(nmask*cellcolor_nuc[1])
        nuc_mask[:,:,2]+=(nmask*cellcolor_nuc[2])
    imwrite(os.path.join(saveDir,mem_maskName),mem_mask.astype(np.uint8))
    imwrite(os.path.join(saveDir,nuc_maskName),nuc_mask.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='BC_CODEX',help='')
    parser.add_argument('--nuc_pred_dir',type=str,default='nucleus_segmentations_WS',help='')
    parser.add_argument('--cell_pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    parser.add_argument('--catcsv',type=str,default='classify_by_dilation',help='')
    parser.add_argument('--save_dir',type=str,default='Composite_predictions-withSAMclass',help='')
    parser.add_argument('--max_cpu',type=int,default=25,help='')

    args,unparsed = parser.parse_known_args()

    cellpredDir = os.path.join(args.rootdir,args.dataset,args.cell_pred_dir)
    nucpredDir = os.path.join(args.rootdir,args.dataset,args.nuc_pred_dir)
    csvDir = os.path.join(args.rootdir,args.dataset,args.catcsv)
    saveDir = os.path.join(args.rootdir,args.dataset,args.save_dir)
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(os.path.join(saveDir,'downsampled')):
        os.makedirs(os.path.join(saveDir,'downsampled'))
    
    csvname = 'combined_SAM-scores_FinalCats.csv'
    print(csvname)
    catcsv = pd.read_csv(os.path.join(csvDir,csvname),low_memory=False)
    print(catcsv.columns)

    memnames = list(np.unique(catcsv['MemCatName']))
    nucnames = list(np.unique(catcsv['NucCatName']))
    
    #save colorbars
    memcolor_dict,memname_dict = make_membrane_colorbar(memnames,saveDir,args.dataset,os.path.join(args.rootdir,args.dataset))
    memcolor_dict['unclassified']=[128,128,128]
    nuccolor_dict,nucname_dict = make_nucleus_colorbar(nucnames,saveDir,args.dataset,os.path.join(args.rootdir,args.dataset))
    make_full_colorbar(memnames,nucnames,memcolor_dict,nuccolor_dict,memname_dict,nucname_dict,saveDir)
    
    #color cells by class
    samples = os.listdir(cellpredDir)
    samples.sort()
    for sample in samples:
        if args.sample!='all_samples':
            if sample!=args.sample:
                continue
        areas = os.listdir(os.path.join(cellpredDir,sample))
        areas.sort()
        for area in areas:
            if args.area!='all_areas':
                if area!=args.area:
                    continue
            checkName = sample+'_'+area+'_nucleusScore-classes.tif'
            if os.path.exists(os.path.join(saveDir,checkName)):
                continue
            compdicts = []
            locdict = {}
                
            comps=os.listdir(os.path.join(cellpredDir,sample,area))
            for comp in comps:
                section = comp.split('_')[-1].split('.')[0]
                segs = np.load(os.path.join(cellpredDir,sample,area,comp),allow_pickle=True).item()
                if 'bbox' in segs.keys():
                    mR,_,mC,_ = segs['bbox']
                else:
                    mR=0
                    mC=0
                segs = segs['masks']
                cells = np.unique(segs)
                cells = np.delete(cells,0)
                r,c = np.shape(segs)
                locdict.update({'Size':[r,c]})
                catdf = catcsv[catcsv['FullSection']==sample+'_'+area+'_'+section]
                
                batchsize = int(np.ceil(len(cells)/args.max_cpu))
                for i in tqdm(range(int(np.ceil(args.max_cpu/len(comps))))):
                    cellids = cells[i*batchsize:(i+1)*batchsize]
                    if len(cellids)==0:
                        continue
                    tmp = np.where((segs>=np.min(cellids)) & (segs<=np.max(cellids)),1,0)
                    cellidx = np.where(tmp==1)
                    del tmp
                    minR = np.min(cellidx[0])
                    minC = np.min(cellidx[1])
                    maxR = np.max(cellidx[0])
                    maxC = np.max(cellidx[1])
                    subdf = catdf[catdf['CellID'].isin(cellids)]
                    mem_maskName = sample+'_'+area+'_'+str(i)+'_'+section+'_membrane-classes.tif'
                    nuc_maskName = sample+'_'+area+'_'+str(i)+'_'+section+'_nucleus-classes.tif'
                    locdict.update({str(i):[minR+mR,maxR+mR,minC+mC,maxC+mC]})
                    compdicts.append({'cellPredDir':cellpredDir,'nucPredDir':nucpredDir,'SaveDir':saveDir,
                                      'Sample':sample,'Area':area,'PredName':comp,'MemMaskName':mem_maskName,
                                      'NucMaskName':nuc_maskName,'CatDF':subdf,'MemColorDict':memcolor_dict,
                                      'NucColorDict':nuccolor_dict,'Bbox':[minR+mR,maxR+mR,minC+mC,maxC+mC],
                                      'Section':section})
            num_workers = args.max_cpu
            pool = mp.Pool(num_workers)
            pool.map(map_to_composite,compdicts)
            pool.close()
            
            r,c = locdict['Size']
            mem_full = np.zeros([r,c,3])
            nuc_full = np.zeros([r,c,3])
            
            for compdict in compdicts:
                mempatch = imread(os.path.join(compdict['SaveDir'],compdict['MemMaskName']))
                nucpatch = imread(os.path.join(compdict['SaveDir'],compdict['NucMaskName']))
                rmin,rmax,cmin,cmax = compdict['Bbox']
                if ((rmax-rmin)==0) or ((cmax-cmin)==0):
                    continue
                r,c,_ = np.shape(mempatch)
                if (r==0) or (c==0):
                    continue
                if cmax>c:
                    print('problem with',compdict['MemMaskName'])
                    continue
                mem_full[rmin:rmax,cmin:cmax,:]+=mempatch
                nuc_full[rmin:rmax,cmin:cmax,:]+=nucpatch
            
            mem_maskName = sample+'_'+area+'_membrane-classes.tif'
            nuc_maskName = sample+'_'+area+'_nucleus-classes.tif'
            small_mem_mask = block_reduce(mem_full,block_size=(10,10,1),func=np.max)
            small_nuc_mask = block_reduce(nuc_full,block_size=(10,10,1),func=np.max)
            imwrite(os.path.join(saveDir,'downsampled',mem_maskName),small_mem_mask.astype(np.uint8))
            imwrite(os.path.join(saveDir,'downsampled',nuc_maskName),small_nuc_mask.astype(np.uint8))
            rmfiles = [x['MemMaskName'] for x in compdicts]+[x['NucMaskName'] for x in compdicts]
            [os.remove(os.path.join(saveDir,x)) for x in rmfiles]  
            

if __name__=="__main__":
    main()
