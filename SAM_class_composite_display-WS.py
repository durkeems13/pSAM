import os
import argparse
import numpy as np
import pandas as pd
from tifffile import imread,imwrite
import multiprocessing as mp
from tqdm import tqdm
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def map_to_composite(batches):
    for batch in batches:
        cellPredDir = batch['cellPredDir']
        nucPredDir = batch['nucPredDir']
        sample = batch['Sample']
        area = batch['Area']
        PredName = batch['PredName']
        if 'S15' in PredName:
            continue

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
            continue
        
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
            nmask = np.where(nuc_pred_comp==cell,1,0)
            celldf = catdf[catdf['CellID']==cell]
            
            memcellcats = list(celldf['MemCatName'])
            nuccellcats = list(celldf['NucCatName'])

            memcatscore = list(celldf['MemCatScore'])
            nuccatscore = list(celldf['NucCatScore'])
            
            cellcat_mem = memcellcats[0]
            cellcolor_mem = memcolor_dict[cellcat_mem]
            cellcat_nuc = nuccellcats[0] 
            cellcolor_nuc = nuccolor_dict[cellcat_nuc]
                
            mem_mask[:,:,0]+=(cmask*cellcolor_mem[0])
            mem_mask[:,:,1]+=(cmask*cellcolor_mem[1])
            mem_mask[:,:,2]+=(cmask*cellcolor_mem[2])
            
            nuc_mask[:,:,0]+=(nmask*cellcolor_nuc[0])
            nuc_mask[:,:,1]+=(nmask*cellcolor_nuc[1])
            nuc_mask[:,:,2]+=(nmask*cellcolor_nuc[2])
                
            mem_score_mask[:,:,0]+=(cmask*cellcolor_mem[0])*memcatscore
            mem_score_mask[:,:,1]+=(cmask*cellcolor_mem[1])*memcatscore
            mem_score_mask[:,:,2]+=(cmask*cellcolor_mem[2])*memcatscore
            
            nuc_score_mask[:,:,0]+=(nmask*cellcolor_nuc[0])*nuccatscore
            nuc_score_mask[:,:,1]+=(nmask*cellcolor_nuc[1])*nuccatscore
            nuc_score_mask[:,:,2]+=(nmask*cellcolor_nuc[2])*nuccatscore
                
        imwrite(os.path.join(saveDir,mem_maskName),mem_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir,nuc_maskName),nuc_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir,memscore_maskName),mem_score_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir,nucscore_maskName),nuc_score_mask.astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='Lupus_Nephritis',help='')
    parser.add_argument('--nuc_pred_dir',type=str,default='nucleus_segmentations_WS',help='')
    parser.add_argument('--cell_pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--sample',type=str,default='012523S4',help='')
    parser.add_argument('--area',type=str,default='Area2',help='')
    parser.add_argument('--catcsv',type=str,default='classify_by_dilation/with-MPI',help='')
    parser.add_argument('--save_dir',type=str,default='Composite_predictions-withSAMclass',help='')
    parser.add_argument('--max_cpu',type=int,default=100,help='')

    args,unparsed = parser.parse_known_args()

    cellpredDir = os.path.join(args.rootdir,args.dataset,args.cell_pred_dir)
    nucpredDir = os.path.join(args.rootdir,args.dataset,args.nuc_pred_dir)
    csvDir = os.path.join(args.rootdir,args.dataset,args.catcsv)
    saveDir = os.path.join(args.rootdir,args.dataset,args.save_dir)
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(os.path.join(saveDir,'downsampled')):
        os.makedirs(os.path.join(saveDir,'downsampled'))
    
    csvname = os.listdir(csvDir)
    csvname = [x for x in csvname if x.endswith('filtered.csv')]
    if len(csvname)>1:
        if (args.sample=='all_samples') and (args.area=='all_areas'):
            csvname=[x for x in csvname if 'combined' in x]
        else:
            csvname = [x for x in csvname if args.sample in x]
            csvname = [x for x in csvname if args.area in x]
    catcsv = pd.read_csv(os.path.join(csvDir,csvname[0]))
    catcsv['MemCatScore']=catcsv['MemCatScore']/catcsv['MemCatScore'].max()
    catcsv['NucCatScore']=catcsv['NucCatScore']/catcsv['NucCatScore'].max()

    memcats = np.unique(catcsv['MemCat'])
    nuccats = np.unique(catcsv['NucCat'])
    memnames = np.unique(catcsv['MemCatName'])
    nucnames = list(np.unique(catcsv['NucCatName']))
    
    memcolor_dict = {}
    memname_dict = {}
    
    if args.dataset=='PSC_CRC':
        #PSC panel
        memcolordict={'APC':[155,5,255],'ActivatedCD8T':[10,242,14],'B':[28,25,227],
                   'CytotoxicT':[6,140,8],'HelperT':[237,14,29],'IL17-HelperT':[166,3,3],
                   'IL17CD45':[247,7,235],'IgG-Plasma':[9,237,188],'OtherT':[246,250,5],
                   'OtherCD45':[123,7,116],'Plasma':[10,166,132],'Plasmablast':[45,148,196],
                   'unclassified':[168,167,165], 'structure':[133,113,60]}
        mem_names = ['APC','GZB+ CD8T cell','B cell','CD8T cell','CD4T cell','IL17+ CD4T cell','IL17+ Other Immune cell','IgG+ Plasma cell',
                    'Other T cell','Other Immune cell','Plasma cell','Plasmablast','Unclassified','Structural cell']
        memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}
    else:
        #kidney panel
        memcolordict={'B':[28,25,227],'B-TRM':[74,131,237],'BowmansCapsule':[117,74,8],'CD4CD8T':[246,250,5],
                   'CD4T':[237,14,29],'CD4T-ICOS':[166,3,3],'CD4T-ICOSPD1':[245,69,69],'CD4T-PD1':[150,47,47],'CD4T-TRM':[219,116,116],
                   'CD8T':[10,242,14],'CD8T-Ex':[6,140,8],'CD8T-GZB':[69,153,70],'CD8T-TRM':[98,166,70],
                   'DistalTubule':[133,113,60],'Endothelial cell':[186,174,141],'GDT':[247,7,235],'HealthyTubule':[176,136,4],'Glom':[148,123,43],
                   'M2':[71,204,180],'Macrophage-CD14+':[61,245,242],'Macrophage-CD16+':[3,171,168],'Monocyte-GZB':[157,227,245],
                   'Monocyte-HLAII+':[0,167,245],'Monocyte-HLAII-':[45,148,196],'NK':[236,245,157],'NKT':[245,210,157],'Neutrophils':[237,120,9],
                   'Plasma':[9,237,188],'Plasma-Act':[10,166,132],'Plasmablast':[59,148,106],'ProximalTubule':[150,142,117],'unclassified':[168,167,165],
                   'cDC1':[155,5,255],'cDC2':[100,32,145],'pDC':[191,121,237]}
        
        mem_names = ['B cell','TRM B cell','Stressed tubule cell','CD4CD8T cell','CD4T cell','ICOS+ CD4T cell','ICOS+PD1+ CD4T cell','PD1+ CD4T cell',
                     'TRM CD4T cell','CD8T cell','Exhausted CD8T cell','GZM+ CD8T cell','TRM CD8T cell','Distal tubule cell','Endothelial cell','\u03B3\u03B4 T cell',
                     'Healthy tubule cell','Glomerulus cell','CD163+ M\u03A6','Other M\u03A6','CD16+ M\u03A6','GZB+ Monocyte','HLAII+ Monocyte',
                     'HLAII- Monocyte','NK cell','NKT cell','Neutrophil','Plasma cell','SLAMF7+ Plasma cell','Plasmablast','Proximal tubule cell',
                     'Unclassified','cDC1','cDC2','pDC']
        memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}

    disp_memnames=[]
    for i,memcat in enumerate(memcats):
        mname = memnames[i]
        val = memcolordict[mname]
        memcolor_dict.update({memcat:val})
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
    plt.savefig(os.path.join(saveDir,'membrane_colorbar.png'))
    #plt.show()
    plt.close()
    
    nuccolor_dict={}
    nucname_dict={}
    #selected nucleus colors
    if args.dataset=='PSC_CRC':
        nuccolordict = {'Nucleus':[200,200,200],
                     'RegT-Nuc':[135,16,88],'pSTAT3-Nuc':[252,152,3]}
        nuc_names = ['Generic Nucleus','FoxP3+ Nucleus','pSTAT3+ Nucleus']
        nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolordict.keys()))}
    else:
        nuccolordict = {'Nucleus':[200,200,200],'Nucleus-FoxP3+':[135,16,88],
                     'Nucleus-Ki67+':[2,94,37],'Nucleus-Tubule':[179,144,4]}
        nuc_names = ['Generic Nucleus','FoxP3+ Nucleus','Ki67+ Nucleus','Tubule Nucleus']
        nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolordict.keys()))}

    disp_nucnames=[]
    for i,nuccat in enumerate(nuccats):
        nname = nucnames[i]
        val = nuccolordict[nname]
        nuccolor_dict.update({nuccat:val}) 
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
    plt.savefig(os.path.join(saveDir,'nucleus_colorbar.png'))
    #plt.show()
    plt.close() 
    
    disp_names=[]
    color_dict={}
    for i,memcat in enumerate(memcats):
        mname = memnames[i]
        val = memcolordict[mname]
        color_dict.update({memcat:val})
        disp_names.append(memname_dict[mname])
    for i,nuccat in enumerate(nuccats):
        nname = nucnames[i]
        val = nuccolordict[nname]
        color_dict.update({nuccat+memcat+1:val}) 
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
    plt.savefig(os.path.join(saveDir,'full_colorbar.png'))
    #plt.show()
    plt.close() 
    
    samples = os.listdir(cellpredDir)
    compdicts = []
    locdict = {}
    for sample in samples:
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue
        if 'S15' in sample:
            continue
        areas = os.listdir(os.path.join(cellpredDir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area not in args.area:
                    continue
            comp=os.listdir(os.path.join(cellpredDir,sample,area))
            comp = comp[0]
            segs = np.load(os.path.join(cellpredDir,sample,area,comp),allow_pickle=True).item()
            segs = segs['masks']
            cells = np.unique(segs)
            cells = np.delete(cells,0)
            
            r,c = np.shape(segs)
            locdict.update({'Size':[r,c]})
            catdf = catcsv[catcsv['CompName']==sample+'_'+area]
            batchsize = int(np.ceil(len(cells)/100))
            for i in tqdm(range(100)):
                cellids = cells[i*batchsize:(i+1)*batchsize]
                tmp = np.where((segs>=np.min(cellids)) & (segs<=np.max(cellids)),1,0)
                cellidx = np.where(tmp==1)
                del tmp
                minR = np.min(cellidx[0])
                minC = np.min(cellidx[1])
                maxR = np.max(cellidx[0])
                maxC = np.max(cellidx[1])
                subdf = catdf[catdf['CellID'].isin(cellids)]
                mem_maskName = sample+'_'+area+'_'+str(i)+'_membrane-classes.tif'
                nuc_maskName = sample+'_'+area+'_'+str(i)+'_nucleus-classes.tif'
                locdict.update({str(i):[minR,maxR,minC,maxC]})
                compdicts.append({'cellPredDir':cellpredDir,'nucPredDir':nucpredDir,'SaveDir':saveDir,
                                  'Sample':sample,'Area':area,'PredName':comp,'MemMaskName':mem_maskName,
                                  'NucMaskName':nuc_maskName,'CatDF':subdf,'MemColorDict':memcolordict,
                                  'NucColorDict':nuccolordict,'Bbox':[minR,maxR,minC,maxC]})
                
    if len(compdicts) > args.max_cpu:
        num_workers = args.max_cpu
    else:
        num_workers = len(compdicts)

    batchsize = int(np.ceil(len(compdicts)/num_workers))
    batches = []
    for i in range(num_workers):
        batches.append(compdicts[i*batchsize:(i+1)*batchsize])
    
    pool = mp.Pool(num_workers)
    pool.map(map_to_composite,batches)
    pool.close()
    
    r,c = locdict['Size']
    mem_full = np.zeros([r,c,3])
    nuc_full = np.zeros([r,c,3])
    memscore_full = np.zeros([r,c,3])
    nucscore_full = np.zeros([r,c,3])
    for compdict in compdicts:
        mempatch = imread(os.path.join(compdict['SaveDir'],compdict['MemMaskName']))
        nucpatch = imread(os.path.join(compdict['SaveDir'],compdict['NucMaskName']))
        memscorepatch = imread(os.path.join(compdict['SaveDir'],compdict['MemMaskName'].replace('.tif','_score.tif')))
        nucscorepatch = imread(os.path.join(compdict['SaveDir'],compdict['NucMaskName'].replace('.tif','_score.tif')))
        rmin,rmax,cmin,cmax = compdict['Bbox']
        if cmax>c:
            print('problem with',compdict['MemMaskName'])
            continue
        mem_full[rmin:rmax,cmin:cmax,:]+=mempatch
        nuc_full[rmin:rmax,cmin:cmax,:]+=nucpatch
        memscore_full[rmin:rmax,cmin:cmax,:]+=memscorepatch
        nucscore_full[rmin:rmax,cmin:cmax,:]+=nucscorepatch
    
    mem_maskName = args.sample+'_'+args.area+'_membrane-classes.tif'
    nuc_maskName = args.sample+'_'+args.area+'_nucleus-classes.tif'
    small_mem_mask = block_reduce(mem_full,block_size=(4,4,1),func=np.max)
    small_nuc_mask = block_reduce(nuc_full,block_size=(4,4,1),func=np.max)
    imwrite(os.path.join(saveDir,mem_maskName),mem_full.astype(np.uint8))
    imwrite(os.path.join(saveDir,nuc_maskName),nuc_full.astype(np.uint8))
    imwrite(os.path.join(saveDir,'downsampled',mem_maskName),small_mem_mask.astype(np.uint8))
    imwrite(os.path.join(saveDir,'downsampled',nuc_maskName),small_nuc_mask.astype(np.uint8))
    
    memscore_maskName = args.sample+'_'+args.area+'_membraneScore-classes.tif'
    nucscore_maskName = args.sample+'_'+args.area+'_nucleusScore-classes.tif'
    small_mem_mask = block_reduce(memscore_full,block_size=(4,4,1),func=np.max)
    small_nuc_mask = block_reduce(nucscore_full,block_size=(4,4,1),func=np.max)
    imwrite(os.path.join(saveDir,memscore_maskName),memscore_full.astype(np.uint8))
    imwrite(os.path.join(saveDir,nucscore_maskName),nucscore_full.astype(np.uint8))
    imwrite(os.path.join(saveDir,'downsampled',memscore_maskName),small_mem_mask.astype(np.uint8))
    imwrite(os.path.join(saveDir,'downsampled',nucscore_maskName),small_nuc_mask.astype(np.uint8))
    
    rmfiles = [x['MemMaskName'] for x in compdicts]+[x['NucMaskName'] for x in compdicts]
    [os.remove(os.path.join(saveDir,x)) for x in rmfiles]  
    rmfiles = [x['MemMaskName'].replace('.tif','_score.tif') for x in compdicts]+[x['NucMaskName'].replace('.tif','_score.tif') for x in compdicts]
    [os.remove(os.path.join(saveDir,x)) for x in rmfiles]  
    

if __name__=="__main__":
    main()
