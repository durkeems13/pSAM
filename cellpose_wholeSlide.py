import os
import argparse
from shutil import copyfile
import numpy as np
from cellpose import utils,io,models
from random import shuffle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='Antibody_Mediated_Rejection',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    parser.add_argument('--readdir',type=str,default='Corrected_DAPI_composites',help='')
    parser.add_argument('--segdir',type=str,default='tissue_composite_masks_instance',help='')
    parser.add_argument('--savedir',type=str,default='segmentations_WS',help='')
    parser.add_argument('--NucOrCell',type=str,default='nucleus',help='')
    parser.add_argument('--seg_model',type=str,default='custom',help='')

    args,unparsed = parser.parse_known_args()
    
    if args.dataset =='CosMx_CODEX':
        dset='Lupus_Nephritis'
    elif args.dataset == 'Normal_Kidney':
        dset='Lupus_Nephritis'
    elif args.dataset == 'Antibody_Mediated_Rejection':
        dset='Renal_Allograft'
    else:
        dset = args.dataset
    
    if args.NucOrCell == 'nucleus':
        rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
        sdir = os.path.join(args.rootdir,args.dataset,args.NucOrCell+'_'+args.savedir)
        if args.seg_model=='custom':
            model = models.CellposeModel(gpu=True,pretrained_model=os.path.join('CustomCPModels','CellPose-FT_'+dset))
        else:
            model = models.CellposeModel(gpu=True,model_type='nuclei')
    if args.NucOrCell=='cell':
        rdir = os.path.join(args.rootdir,args.dataset,args.stackdir)
        sdir = os.path.join(args.rootdir,args.dataset,args.NucOrCell+'_'+args.savedir)
        if args.seg_model=='custom':
            model = models.CellposeModel(gpu=True,pretrained_model=os.path.join('CustomCPModels','CellPose-FT_wholeCell_'+dset))
        else:
            model = models.CellposeModel(gpu=True,model_type='cyto2')
    
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    ims = os.listdir(rdir)
    for im in ims:
        if args.readdir!='aligned_HE':
            try:
                _,_,_,_,sample,area = im.split('_')
            except:
                print(im)
                continue
        else:
            sample,area = im.split('_')
        if 'TMA' in sample:
            continue
        area = area.split('.')[0]
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        if args.area!='all_areas':
            if args.area!=area:
                continue

        if not os.path.exists(os.path.join(sdir,sample)):
            os.makedirs(os.path.join(sdir,sample))
        if not os.path.exists(os.path.join(sdir,sample,area)):
            os.makedirs(os.path.join(sdir,sample,area))
        
        copyfile(os.path.join(rdir,im),os.path.join(sdir,sample,area,im))
        if args.NucOrCell=='nucleus':
            channels=[0,0]
        elif args.NucOrCell=='cell':
            channels=[2,1]
        fname = os.path.join(sdir,sample,area,im)
        try:
            img = io.imread(fname)
        except:
            print(fname,'cannot read in image')
            continue  
        
        from tifffile import imread
        if os.path.exists(os.path.join(args.rootdir,args.dataset,args.segdir)):
            section_masks = os.listdir(os.path.join(args.rootdir,args.dataset,args.segdir))
            section_masks = [x for x in section_masks if (sample in x) and (area in x)]
            if len(section_masks)==0:
                print(sample,area,'Instance segmentations of tissue sections not complete. Please run.')
                continue
            section_masks = imread(os.path.join(args.rootdir,args.dataset,args.segdir,section_masks[0]))
            vals = np.unique(section_masks)
            vals = np.delete(vals,0)
        else: 
            print('Instance segmentations of tissue sections not complete. Please run.')
            continue
        print(vals)
        for val in vals:
            print('Segmenting',fname,val,'...')
            new_img = np.where(section_masks==val,img,0)
            idx = np.where(new_img>0)
            print('***',len(idx[0]))
            if len(idx[0])==0:
                np.save(os.path.join(sdir,sample,area,im.replace('.tif','_seg'+str(val)+'.npy')),{'masks':np.zeros(np.shape(img)),'bbox':(0,0,0,0)},allow_pickle=True)
            rmin = np.min(idx[0])
            rmax = np.max(idx[0])
            cmin = np.min(idx[1])
            cmax = np.max(idx[1])
            new_img = new_img[rmin:rmax,cmin:cmax]
            try:
                masks,_,_ = model.eval(new_img,diameter=None,channels=channels,model_loaded=True,flow_threshold=0,batch_size=1)
            except:
                print('Memory error for', sample, area, val,'. Continuing...')
                continue
            M = np.max(masks,axis=None)
            print(M)
            if M > 65535:
                parts = int(np.ceil(M/65535))
                labels = ['a','b','c','d','e']
                for ii in range(parts):
                    newM = (ii+1)*65535
                    m = (ii*65535)
                    newmasks = np.where((masks<=newM) & (masks>m),masks-m,0)
                    print('Section',ii,np.min(newmasks,axis=None),np.max(newmasks,axis=None))
                    np.save(os.path.join(sdir,sample,area,im.replace('.tif','_seg'+str(val)+labels[ii]+'.npy')),{'masks':newmasks.astype(np.uint16),'bbox':(rmin,rmax,cmin,cmax)},allow_pickle=True)
            else:
                np.save(os.path.join(sdir,sample,area,im.replace('.tif','_seg'+str(val)+'.npy')),{'masks':masks,'bbox':(rmin,rmax,cmin,cmax)},allow_pickle=True)
        os.remove(os.path.join(sdir,sample,area,im))

if __name__=='__main__':
    main()

