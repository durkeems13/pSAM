import os,argparse
import numpy as np
import pandas as pd
from random import shuffle
from tifffile import imread,imwrite
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import threshold_multiotsu,threshold_otsu
from tqdm import tqdm
import multiprocessing as mp

def calc_sim_to_ref(batches):
    for batch in batches:
        read_dir = batch['ImPath']
        sample = batch['Sample']
        area = batch['Area']
        N_ref = batch['NRef']
        reference_spectra = batch['RefDf']
        cols = batch['Cats']
        save_dir = batch['SaveDir']
        masks = batch['MaskList']
        mask_dir = batch['MaskDir']
        dapis = batch['DAPIList']
        dapi_dir = batch['DAPIDir']
        minR,maxR,minC,maxC = batch['Bbox']
        batchnum = batch['BatchNum']
        comps = os.listdir(os.path.join(read_dir,sample,area))
        stains = reference_spectra['Marker']
        stains = list(stains)
        cols = [x for x in cols if x!='Marker']
        reference_spectra = reference_spectra[cols]

        dapiname = [x for x in dapis if (sample in x) and (area in x)]
        if len(dapiname)==0:
            continue
        
        skip_batch=0
        comps_done=[]
        for col in cols:
            if os.path.exists(os.path.join(save_dir,sample,area,'_'.join([sample,area,col,'similarity.tif']))):
                comps_done.append(col)
            if os.path.exists(os.path.join(save_dir,sample,area,'_'.join([sample,area,col,str(batchnum),'similarity.tif']))):
                skip_batch+=1
        if skip_batch==len(cols):
            continue

        cols = list(set(cols)-set(comps_done))
        if len(cols)==0:
            continue
        
        print('')
        print('Stacking composites...')
        if os.path.exists(os.path.join(save_dir,sample,area,'_'.join([sample,area,str(batchnum),'compstack.tif']))):
            imstack = imread(os.path.join(save_dir,sample,area,'_'.join([sample,area,str(batchnum),'compstack.tif'])))
            imstack = imstack/255
            r,c,ch = np.shape(imstack)
        else:
            print('Stacking',sample,area)
            r = maxR-minR
            c = maxC-minC

            new_comps=[]
            for stain in stains:
                new_comp = [x for x in comps if x.split('_')[0]==stain]
                if len(new_comp)==1:
                    new_comps.append(new_comp[0])
                else:
                    new_comps.append('ZERO_MAT')

            imstack = np.zeros([r,c,len(stains)])
            for i,comp in tqdm(enumerate(new_comps)):
                if comp == 'ZERO_MAT':
                    img = np.zeros([maxR-minR,maxC-minC])
                    imstack[:,:,i]=img
                else:
                    img = imread(os.path.join(read_dir,sample,area,comp))
                    tim = [x for x in masks if (sample in x) and (area in x)]
                    timg = imread(os.path.join(mask_dir,tim[0]))
                    bkgd = img[timg==0]
                    m =  np.mean(bkgd)
                    del timg,tim,bkgd
                    iM=np.max(img,axis=None)
                    img = np.clip(img,m,iM)
                    img = img/iM
                    imstack[:,:,i]=img[minR:maxR,minC:maxC]
            if 'DAPI' in stains:
                dapi = imread(os.path.join(dapi_dir,dapiname[0]))
                vals = dapi.flatten()
                vals.sort()
                idx = int(0.999*len(dapi.flatten()))
                dM = vals[idx]
                dapi=np.clip(dapi,0,dM)
                dapi=dapi/dM
                dloc = stains.index('DAPI')
                try:
                    imstack[:,:,dloc]=dapi[minR:maxR,minC:maxC]
                except:
                    imstack[:,:-1,dloc]=dapi[minR:maxR,minC:maxC]
            
            imwrite(os.path.join(save_dir,sample,area,'_'.join([sample,area,str(batchnum),'compstack.tif'])),(255*imstack).astype(np.uint8))
        print('')
        print('Computing SAM maps...')
        locs = np.isnan(imstack)
        imstack[locs]=0
        for ii,ref in tqdm(enumerate(cols)):
            if os.path.exists(os.path.join(save_dir,sample,area,'_'.join([sample,area,ref,str(batchnum),'similarity.tif']))):
                continue
            R = reference_spectra[ref]
            spectral_class=cosine_similarity(imstack.reshape(r*c,-1),np.array(R).reshape(1,-1)).reshape(r,c)
            imwrite(os.path.join(save_dir,sample,area,'_'.join([sample,area,ref,str(batchnum),'similarity.tif'])),(255*spectral_class).astype(np.uint8))
        del imstack
        os.remove(os.path.join(save_dir,sample,area,'_'.join([sample,area,str(batchnum),'compstack.tif'])))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help="")
    parser.add_argument("--dataset",type=str,default='BC_CODEX',help='')
    parser.add_argument("--read",type=str,default='Normalized_composites',help="")
    parser.add_argument("--dapi",type=str,default='Corrected_DAPI_composites',help="")
    parser.add_argument("--mask",type=str,default='tissue_composite_masks_instance',help="")
    parser.add_argument("--save",type=str,default='spectral_angle_mapping',help="")
    parser.add_argument("--sample",type=str,default='all_samples',help="")
    parser.add_argument("--area",type=str,default='all_areas',help="")

    args, unparsed = parser.parse_known_args()
    
    read_dir=os.path.join(args.rootdir,args.dataset,args.read)
    mask_dir=os.path.join(args.rootdir,args.dataset,args.mask)
    dapi_dir=os.path.join(args.rootdir,args.dataset,args.dapi)
    save_dir=os.path.join(args.rootdir,args.dataset,args.save)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir,'class_maps')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reference_spectra = pd.read_csv(os.path.join(args.rootdir,args.dataset,'reference_spectra.csv'))
    cols = reference_spectra.columns
    cols = [x for x in cols if 'Unnamed' not in x]
    #cols = [x for x in cols if 'Nuc' in x]
    N_ref = len(cols)
    samples = os.listdir(read_dir)
    masks = os.listdir(mask_dir)
    dapis = os.listdir(dapi_dir)
    
    compdicts = []
    loc_dict={}
    for sample in samples:
        if 'TMA' in sample:
            continue
        if args.sample!='all_samples':
            if sample != args.sample:
                continue
        areas = os.listdir(os.path.join(read_dir,sample))
        if not os.path.exists(os.path.join(save_dir,sample)):
            os.makedirs(os.path.join(save_dir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area != args.area:
                    continue
            if not os.path.exists(os.path.join(save_dir,sample,area)):
                os.makedirs(os.path.join(save_dir,sample,area))
            ims = os.listdir(os.path.join(read_dir,sample,area))
                   
            img = imread(os.path.join(read_dir,sample,area,ims[0]))
            r,c = np.shape(img)
            locdict={'Size':[r,c]}
            del img
            R = 10
            rpix = int(np.ceil(r/R))
            C = 10
            cpix = int(np.ceil(c/C))
            rstarts = list(np.arange(0,r,rpix))
            cstarts = list(np.arange(0,c,cpix))
            rends = [x+rpix if (x+rpix)<=r else r for x in rstarts]
            cends = [x+cpix if (x+cpix)<=c else c for x in cstarts]
            bboxes = [[x,rends[i],y,cends[j]] for i,x in enumerate(rstarts) for j,y in enumerate(cstarts)]
            bnums = [str(i) for i in range(len(bboxes))]
            locs = {bnums[i]:bboxes[i] for i in range(len(bnums))}
            locdict.update(locs)
            newdicts = [{'ImPath':read_dir,'Sample':sample,'Area':area,'NRef':N_ref,
                         'RefDf':reference_spectra,'Cats':cols,'SaveDir':save_dir,
                         'MaskList':masks,'MaskDir':mask_dir,'DAPIList':dapis,
                         'DAPIDir':dapi_dir,'Bbox':bboxes[i],
                         'BatchNum':x} for i,x in enumerate(bnums)]
            compdicts.extend(newdicts)
            loc_dict.update({sample+'_'+area:locdict})
    
    if len(compdicts)<=50:
        num_workers = len(compdicts)
    else:
        num_workers=50
    batchsize=int(np.ceil(len(compdicts)/50))
    batches = []
    for i in range(num_workers):
        batch = compdicts[i*batchsize:(i+1)*batchsize]
        batches.append(batch)
    
    pool = mp.Pool(num_workers)
    print('')
    print(num_workers)
    print('Calculating...')
    pool.map(calc_sim_to_ref,batches)
    pool.close()
    
    for sample in samples:
        if args.sample!='all_samples':
            if sample != args.sample:
                continue
        areas = os.listdir(os.path.join(read_dir,sample))
        if not os.path.exists(os.path.join(save_dir,sample)):
            os.makedirs(os.path.join(save_dir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area != args.area:
                    continue
            if not os.path.exists(os.path.join(save_dir,sample,area)):
                os.makedirs(os.path.join(save_dir,sample,area))
            maps = os.listdir(os.path.join(save_dir,sample,area))
            maps = [x for x in maps if len(x.split('_'))==5]
            if len(maps)==0:
                continue
            r,c = loc_dict[sample+'_'+area]['Size']
            for col in cols:
                if col=='Marker':
                    continue
                newname = '_'.join([sample,area,col,'similarity.tif'])
                patches = [x for x in maps if x.split('_')[2]==col]
                if os.path.exists(os.path.join(save_dir,sample,area,newname)):
                    if len(patches)==0:
                        continue
                    else:
                        [os.remove(os.path.join(save_dir,sample,area,x)) for x in patches]
                        continue
                full_map = np.zeros([r,c])
                
                keep=0
                for patch in patches:
                    bnum = patch.split('_')[-2]
                    try:
                        rmin,rmax,cmin,cmax = loc_dict[sample+'_'+area][bnum]
                        tmp = imread(os.path.join(save_dir,sample,area,patch))
                        full_map[rmin:rmax,cmin:cmax]=tmp
                    except:
                        keep=1 
                        break
                newname = '_'.join([sample,area,col,'similarity.tif'])
                if keep==0:
                    imwrite(os.path.join(save_dir,sample,area,newname),full_map.astype(np.uint8))
                    print(newname, 'Done')
                    [os.remove(os.path.join(save_dir,sample,area,x)) for x in patches]
                else:
                    print('Problem with mapping',sample,area,'back to global space.')
        

if __name__=="__main__":
    main()
