import os,argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from skimage.transform import resize
from tifffile import imread,imwrite
import multiprocessing as mp

def interstitial_analyses(batch):
    save_dir = batch['SaveDir']
    sam_dir = batch['SAMDir']
    sample = batch['Sample']
    area = batch['Area']
    catdf = batch['CatDF']
    cellseg_dir = batch['CellSegDir']
    cellseg_name = batch['CellSegName']
    tseg_dir = batch['TissueSegDir']
    cat_list_nuc = batch['NucCatDict']
    cat_list_mem = batch['MemCatDict']
    cat_list_all = batch['CatDict']
    compartment = batch['Compartment']
    if not os.path.exists(os.path.join(save_dir,'viz')):
        os.makedirs(os.path.join(save_dir,'viz'))
    if not os.path.exists(os.path.join(save_dir,'hists')):
        os.makedirs(os.path.join(save_dir,'hists'))
    print('Loading images...')
    if compartment=='Mem':
        classname='memMaxClass.tif'
        scorename='memMaxScore.tif'
        cat_list=cat_list_mem
    elif compartment=='Nuc':
        classname='nucMaxClass.tif'
        scorename='nucMaxScore.tif'
        cat_list=cat_list_nuc
    else:
        classname='maxClass.tif'
        scorename='maxScore.tif'
        cat_list=cat_list_all

    tsegname = os.listdir(tseg_dir)
    tsegname = [x for x in tsegname if (sample in x) and (area in x)]
    maxclass = imread(os.path.join(sam_dir,sample,area,'_'.join([sample,area,classname])))
    maxscore = imread(os.path.join(sam_dir,sample,area,'_'.join([sample,area,scorename])))
    segs = np.load(os.path.join(cellseg_dir,sample,area,cellseg_name),allow_pickle=True).item()
    segs = segs['masks'] 
    tsegs = imread(os.path.join(tseg_dir,tsegname[0]))
   
    r,c = np.shape(tsegs)
    R,C = np.shape(segs)
    if (r!=R) or (c!=C):
        segs = resize(segs,(r,c),anti_aliasing=True,preserve_range=True)

    tsegs = np.where(tsegs>0,1,0)        
    bin_segs = np.where(segs>0,1,0)
    interstitium = np.where((tsegs-bin_segs)==1,1,0)
    maxclass = maxclass*tsegs
    maxscore = maxscore*tsegs    

    interstitial_cats = [[i,x] for i,x in enumerate(cat_list)] # if ('Macrophage' in x) or ('DC' in x)]
    #print(interstitial_cats)
    cat_list2 = cat_list.copy()
    cat_list2.sort()
    ind = [cat_list.index(x) for x in cat_list2]
    interstitial_cats = [interstitial_cats[i] for i in ind]
    #print(interstitial_cats)
    
    print(100*np.sum(interstitium,axis=None)/np.sum(tsegs,axis=None),'% of tissue is outside of cell segs')
    
    if not os.path.exists(os.path.join(save_dir,'median_proximity_scores.npy')):
        prox_mat_mean = np.zeros([len(cat_list),len(cat_list)])
        prox_mat_mean_normCell = np.zeros([len(cat_list),len(cat_list)])
        prox_mat_mean_normPixel = np.zeros([len(cat_list),len(cat_list)])
        prox_mat_median = np.zeros([len(cat_list),len(cat_list)])
        prox_mat_median_normCell = np.zeros([len(cat_list),len(cat_list)])
        prox_mat_median_normPixel = np.zeros([len(cat_list),len(cat_list)])
        
        for jj,mcat in enumerate(cat_list2):
            tmp = catdf[catdf.MemCatName==mcat]
            cellids = tmp.CellID.tolist()      
            cellmask = np.isin(segs,cellids)
            
            dist_from_cell = distance_transform_edt(np.invert(cellmask))
            '''
            # 1/dist from border
            prox_to_cell = (dist_from_cell-np.max(dist_from_cell,axis=None))*-1
            prox_to_cell = prox_to_cell/np.max(prox_to_cell,axis=None)
            prox_to_cell = np.where(cellmask==1,0,prox_to_cell)
            '''
            #gaussian 
            prox_to_cell = np.exp(-0.001*(dist_from_cell**2))
            prox_to_cell = np.where(segs!=0,0,prox_to_cell)
            if not os.path.exists(os.path.join(save_dir,'viz',mcat+'_proximity_to_cell.tif')):
                imwrite(os.path.join(save_dir,'viz',mcat+'_proximity_to_cell.tif'),(255*prox_to_cell/np.max(prox_to_cell,axis=None)).astype(np.uint8))
            '''  
            plt.imshow(prox_to_cell)
            plt.title('Proximity to '+mcat+' cell')
            plt.show()
            '''
        
            for ii,(i,cat) in enumerate(interstitial_cats):
                print(mcat,'cells',cat,'pixels')
                
                if os.path.exists(os.path.join(save_dir,'viz',cat+'_interstitial_pixels.tif')):
                    intcatmask = imread(os.path.join(save_dir,'viz',cat+'_interstitial_pixels.tif'))
                    intcatmask = intcatmask/255
                else:
                    catmask = np.where(maxclass==i,1,0)
                    #catmask = catmask*maxscore
                    catmask = catmask*tsegs
                    intcatmask = catmask*interstitium
                    del catmask
                    imwrite(os.path.join(save_dir,'viz',cat+'_interstitial_pixels.tif'),(255*intcatmask).astype(np.uint8))
                    
                mask_dists = prox_to_cell*intcatmask            
                mask_dists = mask_dists[mask_dists!=0]
                
                prox_mat_mean[ii,jj]=np.mean(mask_dists)
                prox_mat_mean_normCell[ii,jj]=np.mean(mask_dists)/len(cellids)
                prox_mat_mean_normPixel[ii,jj]=np.mean(mask_dists)/len(mask_dists)
                prox_mat_median[ii,jj]=np.median(mask_dists)
                prox_mat_median_normCell[ii,jj]=np.median(mask_dists)/len(cellids)
                prox_mat_median_normPixel[ii,jj]=np.median(mask_dists)/len(mask_dists)
                 
                if not os.path.exists(os.path.join(save_dir,'viz',cat+'_pixels_and_'+mcat+'_cells.tif')):
                    rgb = np.stack([255*cellmask,255*intcatmask,np.zeros(np.shape(cellmask))],axis=2)
                    rgb = block_reduce(rgb,block_size=(5,5,1),func=np.mean)
                    imwrite(os.path.join(save_dir,'viz',cat+'_pixels_and_'+mcat+'_cells.tif'),rgb.astype(np.uint8))
                    rgb = np.stack([255*cellmask,255*intcatmask*prox_to_cell,np.zeros(np.shape(cellmask))],axis=2)
                    rgb = block_reduce(rgb,block_size=(5,5,1),func=np.mean)
                    imwrite(os.path.join(save_dir,'viz',cat+'_weightedPixels_and_'+mcat+'_cells.tif'),rgb.astype(np.uint8))
                
                del intcatmask
                '''
                print('Proximity score for',cat,'to',mcat,':')
                print('Total points:',len(mask_dists))
                print('Mean:',np.mean(mask_dists))
                print('Std:',np.std(mask_dists))
                print('Median:',np.median(mask_dists))
                print('')
                '''
                if not os.path.exists(os.path.join(save_dir,'hists',cat+'_to_'+mcat+'_proximity_scores.png')):
                    plt.hist(mask_dists[mask_dists>0.01],bins=100,label=cat,alpha=0.6,density=True,color='red')
                    plt.xlabel('Proximity Score')
                    plt.title('Proximity scores of '+cat+' pixels to '+mcat+' cells')
                    plt.ylabel('Pixel density')
                    plt.xlim([0,1])
                    #plt.ylim([0,1])
                    #plt.legend(bbox_to_anchor=(1.05,0.5),loc='center left')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir,'hists',cat+'_to_'+mcat+'_proximity_scores.png'),dpi=600)
                    #plt.show()
                    plt.close()  
                
        np.save(os.path.join(save_dir,'median_proximity_scores.npy'),prox_mat_median)
        np.save(os.path.join(save_dir,'Cellnormed_median_proximity_scores.npy'),prox_mat_median_normCell)
        np.save(os.path.join(save_dir,'Pixelnormed_median_proximity_scores.npy'),prox_mat_median_normPixel)
        np.save(os.path.join(save_dir,'mean_proximity_scores.npy'),prox_mat_mean)
        np.save(os.path.join(save_dir,'Cellnormed_mean_proximity_scores.npy'),prox_mat_mean_normCell)
        np.save(os.path.join(save_dir,'Pixelnormed_mean_proximity_scores.npy'),prox_mat_mean_normPixel)
    else:
        prox_mat_median=np.load(os.path.join(save_dir,'median_proximity_scores.npy'))
        prox_mat_median_normCell=np.load(os.path.join(save_dir,'Cellnormed_median_proximity_scores.npy'))
        prox_mat_median_normPixel=np.load(os.path.join(save_dir,'Pixelnormed_median_proximity_scores.npy'))
        prox_mat_mean=np.load(os.path.join(save_dir,'mean_proximity_scores.npy'))
        prox_mat_mean_normCell=np.load(os.path.join(save_dir,'Cellnormed_mean_proximity_scores.npy'),)
        prox_mat_mean_normPixel=np.load(os.path.join(save_dir,'Pixelnormed_mean_proximity_scores.npy'))
    
            
    #make heatmaps of cell/pixel proximity
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_median[:,:],cbar_kws={'label':'Median Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'Median_proximity_score_heatmap.png'),dpi=600)
    plt.show()
                
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_median_normCell[:,:],cbar_kws={'label':'Normalized Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'CellNormalized_median_proximity_score_heatmap.png'),dpi=600)
    plt.show()
    
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_median_normPixel[:,:],cbar_kws={'label':'Normalized Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'PixelNormalized_median_proximity_score_heatmap.png'),dpi=600)
    plt.show()
    
    #make heatmaps of cell/pixel proximity
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_mean[:,:],cbar_kws={'label':'Mean Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'Mean_proximity_score_heatmap.png'),dpi=600)
    plt.show()
                
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_mean_normCell[:,:],cbar_kws={'label':'Normalized Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'CellNormalized_mean_proximity_score_heatmap.png'),dpi=600)
    plt.show()
    
    plt.figure(figsize=(7,7))
    sn.heatmap(prox_mat_mean_normPixel[:,:],cbar_kws={'label':'Normalized Proximity Score'})
    plt.yticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=0,labels=cat_list2,fontsize=9)
    plt.xticks(ticks=np.arange(0,len(cat_list))+0.5,rotation=90,labels=cat_list2,fontsize=9)
    plt.xlabel('Interstitial pixel class',fontsize=11)
    plt.ylabel('Cell class',fontsize=11)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'PixelNormalized_mean_proximity_score_heatmap.png'),dpi=600)
    plt.show()
                              

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help="")
    parser.add_argument("--dataset",type=str,default='Renal_Allograft',help="")
    parser.add_argument("--cellsegs",type=str,default='wholeCell_segmentations_WS',help="")
    parser.add_argument("--tissuesegs",type=str,default='tissue_composite_masks_instance',help="")
    parser.add_argument("--sam",type=str,default='SAM_analysis/MIPs',help="")
    parser.add_argument("--catdir",type=str,default='classify_by_dilation/with-MPI',help="")
    parser.add_argument("--sample",type=str,default='011223S3',help="")
    parser.add_argument("--area",type=str,default='Area2',help="")
    parser.add_argument("--pxsz",type=float,default=0.1507,help="")
    parser.add_argument("--save",type=str,default='SAM_analysis/interstitial_pixel_class',help="")

    args, unparsed = parser.parse_known_args()
    
    cellseg_dir=os.path.join(args.rootdir,args.dataset,args.cellsegs)
    tseg_dir=os.path.join(args.rootdir,args.dataset,args.tissuesegs)
    sam_dir=os.path.join(args.rootdir,args.dataset,args.sam)
    cat_dir=os.path.join(args.rootdir,args.dataset,args.catdir)
    save_dir=os.path.join(args.rootdir,args.dataset,args.save)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reference_spectra = pd.read_csv(os.path.join(args.rootdir,args.dataset,'reference_spectra.csv'))
    cols = reference_spectra.columns
    cols = [x for x in cols if 'Unnamed' not in x]
    cols = [x for x in cols if 'PSTAT' not in x]
    cols = [x for x in cols if 'Bkgd' not in x]
    cols = [x for x in cols if x!='RBCs']
    
    print('Reading master csv...')
    cat_df = pd.read_csv(os.path.join(cat_dir,'combined_SAM-scores_with-MPI_filtered.csv'))
 
    samples = os.listdir(sam_dir)
    samples = [x for x in samples if len(x.split('.'))==1]
    samples.sort()
    
    print('Generating batches for MP...')
    batches=[]
    for sample in samples:
        if sample == '110122S1':
            continue
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        if not os.path.exists(os.path.join(save_dir,sample)):
            os.makedirs(os.path.join(save_dir,sample))
        catdf1 = cat_df[cat_df.Sample==sample]
        areas = os.listdir(os.path.join(sam_dir,sample))
        areas.sort()
        for area in areas:
            if args.area!='all_areas':
                if args.area!=area:
                    continue
            if not os.path.exists(os.path.join(save_dir,sample,area)):
                os.makedirs(os.path.join(save_dir,sample,area))
            catdf2 = catdf1[catdf1.Area==area]
            segnames = os.listdir(os.path.join(cellseg_dir,sample,area))
            cat_dict_mem=pkl.load(open(os.path.join(sam_dir,sample,area,'_'.join([sample,area,'memCatOrder.pkl'])),'rb'))
            cat_dict_nuc=pkl.load(open(os.path.join(sam_dir,sample,area,'_'.join([sample,area,'nucCatOrder.pkl'])),'rb'))
            cat_dict_all=pkl.load(open(os.path.join(sam_dir,sample,area,'_'.join([sample,area,'catOrder.pkl'])),'rb'))
            '''
            if len(segnames)==0:
                continue
            elif len(segnames)==1:
                segname=segnames[0]
            else:
                print(segnames)
                continue
            '''
            segname = segnames[0]
            batches.append({'SaveDir':os.path.join(save_dir,sample,area),'SAMDir':sam_dir,'Sample':sample,
                            'Area':area,'CellSegDir':cellseg_dir,'CellSegName':segname,
                            'TissueSegDir':tseg_dir,'CatDF':catdf2,'Compartment':'Mem',
                            'NucCatDict':cat_dict_nuc,'MemCatDict':cat_dict_mem,'CatDict':cat_dict_all})
    num_workers=50
    if len(batches)<=num_workers:
        num_workers = len(batches)
    else:
        batchsize = int(np.ceil(len(batches)/num_workers))
        new_batches=[]
        for i in range(num_workers):
            new_batches.append(batches[i*batchsize:(i+1)*batchsize])
        batches = new_batches
        del new_batches

    print('')
    print(num_workers)
    pool=mp.Pool(num_workers)
    pool.map(interstitial_analyses,batches)
    pool.close()
    

    
if __name__=="__main__":
    main()
