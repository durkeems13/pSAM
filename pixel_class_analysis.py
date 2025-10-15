#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:14:23 2023

@author: durkeems
"""
import os,argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tifffile import imread,imwrite
from skimage.measure import block_reduce
from skimage.transform import resize
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sn

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def make_color_MIP(mdir,mddir,tseg_dir,cats,mcats,ncats,cdict,comp):
    sample,area = comp.split('_')

    if not os.path.exists(os.path.join(mddir,sample)):
        os.makedirs(os.path.join(mddir,sample))
    
    if not os.path.exists(os.path.join(mddir,sample,area)):
        os.makedirs(os.path.join(mddir,sample,area))
    
    if not os.path.exists(os.path.join(mddir,sample,area,comp+'_colorMIP-Scaled_DS.tif')):
        catimg = imread(os.path.join(mdir,sample,area,comp+'_maxClass.tif'))
        scoreimg = imread(os.path.join(mdir,sample,area,comp+'_maxScore.tif'))
        tims = os.listdir(tseg_dir)
        tim = [x for x in tims if comp in x]
        timg = imread(os.path.join(tseg_dir,tim[0]))
        tR,tC = np.shape(timg)
        R,C = np.shape(catimg)
        if (tR!=R) or (tC!=C):
            timg = np.uint8(resize(timg,(R,C)))
        timg[timg>0]=1
        label1=['Bkgd']
        cats = label1+cats
        catimg+=1
        catimg=catimg*timg
        scoreimg=scoreimg*timg
        r,c = np.shape(catimg)
        rgb_disp = np.zeros([r,c,3],dtype=np.uint8)
        for i,cat in enumerate(cats):
            if cat=='Bkgd':
                continue
            catloc = np.where(catimg==i)
            rgb_disp[catloc[0],catloc[1],:]=cdict[cat]
        #imwrite(os.path.join(mddir,sample,area,comp+'_colorMIP.tif'),rgb_disp)
        rgb2 = rgb_disp.copy()
        rgb2[:,:,0]=np.float32(rgb2[:,:,0])*(scoreimg/255)
        rgb2[:,:,1]=np.float32(rgb2[:,:,1])*(scoreimg/255)
        rgb2[:,:,2]=np.float32(rgb2[:,:,2])*(scoreimg/255)
        #imwrite(os.path.join(mddir,sample,area,comp+'_colorMIP-Scaled.tif'),rgb2.astype(np.uint8))

        rgb3 = block_reduce(rgb2,block_size=(2,2,1),func=np.max)
        imwrite(os.path.join(mddir,sample,area,comp+'_colorMIP-Scaled_DS.tif'),rgb3.astype(np.uint8))

    if not os.path.exists(os.path.join(mddir,sample,area,comp+'_memColorMIP-Scaled_DS.tif')):
        catimg = imread(os.path.join(mdir,sample,area,comp+'_memMaxClass.tif'))
        scoreimg = imread(os.path.join(mdir,sample,area,comp+'_memMaxScore.tif'))
        tims = os.listdir(tseg_dir)
        tim = [x for x in tims if comp in x]
        timg = imread(os.path.join(tseg_dir,tim[0]))
        tR,tC = np.shape(timg)
        R,C = np.shape(catimg)
        if (tR!=R) or (tC!=C):
            timg = np.uint8(resize(timg,(R,C)))
        timg[timg>0]=1
        label1=['Bkgd']
        mcats = label1+mcats
        catimg+=1
        catimg=catimg*timg
        scoreimg=scoreimg*timg
        r,c = np.shape(catimg)
        rgb_disp = np.zeros([r,c,3],dtype=np.uint8)
        for i,cat in enumerate(mcats):
            if cat=='Bkgd':
                continue
            catloc = np.where(catimg==i)
            rgb_disp[catloc[0],catloc[1],:]=cdict[cat]
        #imwrite(os.path.join(mddir,sample,area,comp+'_memColorMIP.tif'),rgb_disp)
        rgb2 = rgb_disp.copy()
        rgb2[:,:,0]=np.float32(rgb2[:,:,0])*(scoreimg/255)
        rgb2[:,:,1]=np.float32(rgb2[:,:,1])*(scoreimg/255)
        rgb2[:,:,2]=np.float32(rgb2[:,:,2])*(scoreimg/255)
        #imwrite(os.path.join(mddir,sample,area,comp+'_memColorMIP-Scaled.tif'),rgb2.astype(np.uint8))
        rgb3 = block_reduce(rgb2,block_size=(2,2,1),func=np.max)
        imwrite(os.path.join(mddir,sample,area,comp+'_memColorMIP-Scaled_DS.tif'),rgb3.astype(np.uint8))
    
    if not os.path.exists(os.path.join(mddir,sample,area,comp+'_nucColorMIP-Scaled_DS.tif')):
        catimg = imread(os.path.join(mdir,sample,area,comp+'_nucMaxClass.tif'))
        scoreimg = imread(os.path.join(mdir,sample,area,comp+'_nucMaxScore.tif'))
        tims = os.listdir(tseg_dir)
        tim = [x for x in tims if comp in x]
        timg = imread(os.path.join(tseg_dir,tim[0]))
        tR,tC = np.shape(timg)
        R,C = np.shape(catimg)
        if (tR!=R) or (tC!=C):
            timg = np.uint8(resize(timg,(R,C)))
        timg[timg>0] = 1
        label1=['Bkgd']
        ncats = label1+ncats
        catimg+=1
        catimg=catimg*timg
        scoreimg=scoreimg*timg
        r,c = np.shape(catimg)
        rgb_disp = np.zeros([r,c,3],dtype=np.uint8)
        for i,cat in enumerate(ncats):
            if cat=='Bkgd':
                continue
            catloc = np.where(catimg==i)
            rgb_disp[catloc[0],catloc[1],:]=cdict[cat]
        #imwrite(os.path.join(mddir,sample,area,comp+'_nucColorMIP.tif'),rgb_disp)
        rgb2 = rgb_disp.copy()
        rgb2[:,:,0]=np.float32(rgb2[:,:,0])*(scoreimg/255)
        rgb2[:,:,1]=np.float32(rgb2[:,:,1])*(scoreimg/255)
        rgb2[:,:,2]=np.float32(rgb2[:,:,2])*(scoreimg/255)
        #imwrite(os.path.join(mddir,sample,area,comp+'_nucColorMIP-Scaled.tif'),rgb2.astype(np.uint8))
        rgb3 = block_reduce(rgb2,block_size=(2,2,1),func=np.max)
        imwrite(os.path.join(mddir,sample,area,comp+'_nucColorMIP-Scaled_DS.tif'),rgb3.astype(np.uint8))
    
def get_class_map_MIPs(batches):
    for batch in batches:
        sample=batch['Sample']
        area = batch['Area']
        sam_dir=batch['SAMDir']
        colordict=batch['ColorDict']
        mip_dir=batch['MIPSaveDir']
        mip_display_dir=batch['MIPDisplayDir']
        tissueseg_dir=batch['TissueSegDir']
        th_dir=batch['thDir']
        C = 1.0

        if not os.path.exists(os.path.join(mip_dir,sample)):
            try:
                os.makedirs(os.path.join(mip_dir,sample))
            except:
                print('Asynchronous directory creation. Continuing...')
        
        if not os.path.exists(os.path.join(mip_dir,sample,area)):
            try:
                os.makedirs(os.path.join(mip_dir,sample,area))
            except:
                print('Asynchronous directory creation. Continuing...')
        
        ims = os.listdir(os.path.join(sam_dir,sample,area))
        if len(ims)==0:
            continue
        cats = [x.split('_')[2] for x in ims]
        nuccats = [x for x in cats if 'Nuc' in x]
        memcats = [x for x in cats if 'Nuc' not in x]
        
        mips = os.listdir(os.path.join(mip_dir,sample,area))
        if len(mips) < 9:
                             
            tmp = imread(os.path.join(sam_dir,sample,area,ims[0]))
            r,c = np.shape(tmp)
            del tmp
            
            [print(x,y) for x,y in zip(ims,cats)]
            imstack = np.zeros([r,c,len(cats)])
            for i,cat in tqdm(enumerate(cats)):
                imname = [x for x in ims if x.split('_')[2]==cat]
                img = imread(os.path.join(sam_dir,sample,area,imname[0]))
                thdict = np.load(os.path.join(th_dir,sample,area,'_'.join([cat,'hist','th.npy'])),allow_pickle=True).item()
                th = thdict['th']
                img = np.where(img<(C*th),0,img)
                imstack[:,:,i]=img
                del img
            maxclass = np.argmax(imstack,axis=-1)
            maxscore = np.max(imstack,axis=-1)
            del imstack
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'maxClass.tif'])),maxclass.astype(np.uint8))
            del maxclass
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'maxScore.tif'])),maxscore.astype(np.uint8))
            del maxscore
            pkl.dump(cats,open(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'catOrder.pkl'])),'wb'))                                              
        
            memstack = np.zeros([r,c,len(memcats)])
            for i,cat in tqdm(enumerate(memcats)):
                imname = [x for x in ims if x.split('_')[2]==cat]
                img = imread(os.path.join(sam_dir,sample,area,imname[0]))
                thdict = np.load(os.path.join(th_dir,sample,area,'_'.join([cat,'hist','th.npy'])),allow_pickle=True).item()
                th = thdict['th']
                img = np.where(img<(C*th),0,img)
                memstack[:,:,i]=img
                del img
            maxclass = np.argmax(memstack,axis=-1)
            maxscore = np.max(memstack,axis=-1)
            del memstack
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'memMaxClass.tif'])),maxclass.astype(np.uint8))
            del maxclass
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'memMaxScore.tif'])),maxscore.astype(np.uint8))
            del maxscore
            pkl.dump(memcats,open(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'memCatOrder.pkl'])),'wb'))
            
            nucstack = np.zeros([r,c,len(nuccats)])
            for i,cat in tqdm(enumerate(nuccats)):
                imname = [x for x in ims if x.split('_')[2]==cat]
                img = imread(os.path.join(sam_dir,sample,area,imname[0]))
                thdict = np.load(os.path.join(th_dir,sample,area,'_'.join([cat,'hist','th.npy'])),allow_pickle=True).item()
                th = thdict['th']
                img = np.where(img<(C*th),0,img)
                nucstack[:,:,i]=img
                del img
            maxclass = np.argmax(nucstack,axis=-1)
            maxscore = np.max(nucstack,axis=-1)
            del nucstack
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'nucMaxClass.tif'])),maxclass.astype(np.uint8))
            del maxclass
            imwrite(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'nucMaxScore.tif'])),maxscore.astype(np.uint8))
            del maxscore
            pkl.dump(nuccats,open(os.path.join(mip_dir,sample,area,'_'.join([sample,area,'nucCatOrder.pkl'])),'wb'))
        print(sample,area,batch['DispComp'])
        if (sample+'_'+area)==batch['DispComp']:
            print('Going to make color MIP')
            make_color_MIP(mip_dir,mip_display_dir,tissueseg_dir,cats,memcats,nuccats,colordict,batch['DispComp'])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help="")
    parser.add_argument("--dataset",type=str,default='BC_CODEX',help="")
    parser.add_argument("--tissuesegs",type=str,default='tissue_composite_masks_instance',help="")
    parser.add_argument("--sam",type=str,default='spectral_angle_mapping/class_maps',help="")
    parser.add_argument("--th",type=str,default='spectral_angle_mapping/map_thresholding',help="")
    parser.add_argument("--sample",type=str,default='all_samples',help="")
    parser.add_argument("--area",type=str,default='all_areas',help="")
    parser.add_argument("--pxsz",type=float,default=0.311,help="")
    parser.add_argument("--save",type=str,default='SAM_analysis',help="")

    args, unparsed = parser.parse_known_args()
    
    tissueseg_dir=os.path.join(args.rootdir,args.dataset,args.tissuesegs)
    sam_dir=os.path.join(args.rootdir,args.dataset,args.sam)
    th_dir=os.path.join(args.rootdir,args.dataset,args.th)
    save_dir=os.path.join(args.rootdir,args.dataset,args.save)
    mip_dir=os.path.join(save_dir,'MIPs')
    mip_display_dir=os.path.join(save_dir,'MIP_display')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(mip_dir):
        os.makedirs(mip_dir)
    if not os.path.exists(mip_display_dir):
        os.makedirs(mip_display_dir)

    samples = os.listdir(sam_dir)
    samples = [x for x in samples if len(x.split('.'))==1]
    batches = []
    for sample in samples:
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        areas = os.listdir(os.path.join(sam_dir,sample))
        for area in areas:
            if args.area!='all_areas':
                if args.area!=area:
                    continue  
            if os.path.exists(os.path.join(args.rootdir,args.dataset,'color_dictionary.npy')):
                colordict = np.load(os.path.join(args.rootdir,args.dataset,'color_dictionary.npy'),allow_pickle=True).item()
                disp_names = [x.replace('-',' ') for x in colordict.keys()]
                dispname_dict = {ky:disp_names[i] for i,ky in enumerate(colordict.keys())}
            else:
                print('***IN WRONG LOCATION***',sample,area)
                if args.dataset=='PSC_CRC':
                    colordict={'APC':[155,5,255],'ActivatedCD8T':[10,242,14],'B':[28,25,227],
                               'CytotoxicT':[6,140,8],'HelperT':[237,14,29],'IL17-HelperT':[166,3,3],
                               'IL17CD45':[247,7,235],'IgG-Plasma':[9,237,188],'OtherT':[246,250,5],
                               'OtherCD45':[123,7,116],'Plasma':[10,166,132],'Plasmablast':[45,148,196],
                               'Unclassified':[168,167,165],'structure':[133,113,60],'Nucleus':[200,200,200],
                               'RegT-Nuc':[135,16,88],'pSTAT3-Nuc':[252,152,3],'OtherIgG':[3,153,161]}
                    disp_names = ['APC','GZB+ CD8T cell','B cell','CD8T cell','CD4T cell','IL-17+ CD4T cell',
                                  'IL17+ Other\nImmune cell','IgG+ Plasma cell','Other T cell','Other Immune cell',
                                  'Plasma cell','Plasmablast','Unclassified','Structural cell','Generic Nucleus',
                                  'Foxp3+ Nucleus','pSTAT3+ Nucleus','Other IgG+ cell']
                    dispname_dict = {ky:disp_names[i] for i,ky in enumerate(colordict.keys())}
                elif args.dataset == 'BC_CODEX':
                    #BC panel
                    colordict={'B-cell':[28,25,227],'CD4T-cell':[237,14,29],'CD8T-cell':[10,242,14],
                                  'EMT':[117,74,8],'Endothelial-cell':[186,174,141],'Exhausted-CD8T-cell':[6,140,8],
                                  'GZB+-CD8T-cell':[69,153,70],'Generic-Tumor':[133,113,60],
                                  'ICOS+-CD4T-cell':[166,3,3],'ICOS+PD1+-CD4T-cell':[245,69,69],
                                  'M1':[71,204,180],'M2':[61,245,242],'Myoepithelial-cell':[176,136,4],
                                  'Neutrophil':[237,120,9],'Other-T-cell':[246,250,5],'PD1+-CD4T-cell':[150,47,47],
                                  'Plasma-cell':[74,131,237],'TAM':[3,171,168],'TH1-cell':[219,116,116],
                                  'mDC':[100,32,145],'pDC':[191,121,237],'unclassified':[168,167,165],
                                  'Foxp3+-Nucleus':[135,16,88],'Generic-Nucleus':[200,200,200],
                                  'Ki76+-Nucleus':[2,94,37]}                
                    disp_names = [x.replace('-',' ') for x in colordict.keys()]
                    disp_names = [x.replace('76','67') for x in colordict.keys()]
                    dispname_dict = {ky:disp_names[i] for i,ky in enumerate(list(colordict.keys()))}
                else:
                    colordict={'B':[28,25,227],'B-TRM':[74,131,237],'BowmansCapsule':[117,74,8],'CD4CD8T':[246,250,5],
                               'CD4T':[237,14,29],'CD4T-ICOS':[166,3,3],'CD4T-ICOSPD1':[245,69,69],'CD4T-PD1':[150,47,47],'CD4T-TRM':[219,116,116],
                               'CD8T':[10,242,14],'CD8T-Ex':[6,140,8],'CD8T-GZB':[69,153,70],'CD8T-TRM':[98,166,70],
                               'DistalTubule':[133,113,60],'Endothelial cell':[186,174,141],'GDT':[247,7,235],'HealthyTubule':[176,136,4],'Glom':[148,123,43],
                               'M2':[71,204,180],'Macrophage-CD14+':[61,245,242],'Macrophage-CD16+':[3,171,168],'Monocyte-GZB':[157,227,245],
                               'Monocyte-HLAII+':[0,167,245],'Monocyte-HLAII-':[45,148,196],'NK':[236,245,157],'NKT':[245,210,157],'Neutrophils':[237,120,9],
                               'Plasma':[9,237,188],'Plasma-Act':[10,166,132],'Plasmablast':[59,148,106],'ProximalTubule':[150,142,117],'Unclassified':[168,167,165],
                               'cDC1':[155,5,255],'cDC2':[100,32,145],'pDC':[191,121,237],
                               'Nucleus':[200,200,200],'Nucleus-FoxP3+':[135,16,88],
                               'Nucleus-Ki67+':[2,94,37],'Nucleus-Tubule':[179,144,4],'SlideBkgd':[25,25,25],'TissueBkgd':[225,225,225]}
                    disp_names = ['B cell','TRM B cell','Bowmans Capsule','CD4CD8T cell','CD4T cell','ICOS+ CD4T cell','ICOS+PD1+ CD4T cell',
                                  'PD1+ CD4T cell','TRM CD4 cell','CD8T cell','Exhausted CD8T cell','GZB+ CD8T cell','TRM CD8T cell','Distal tubule',
                                  'Endothelial cell','Gamma-Delta T cell','Healthy tubule','Glomerulus','CD163+ Macrophage','CD14+ Macrophage','CD16+ Macrophage',
                                  'GZB+ Monocyte','HLAII+ Monocyte','HLAII- Monocyte','NK cell','NKT cell','Neutrophil','Plasma cell','SLAMF7+ Plasma cell',
                                  'Plasmablast','Proximal tubule','Unclassified','cDC1','cDC2','pDC','Generic Nucleus','Foxp3+ Nucleus','Ki67+ Nucleus','Tubule Nucleus','Slide Background','Tissue Background']
                dispname_dict = {ky:disp_names[i] for i,ky in enumerate(colordict.keys())}
            dcomp=sample+'_'+area
            batch_dict = {'Sample':sample,'Area':area,'SAMDir':sam_dir,'ColorDict':colordict,
                          'MIPSaveDir':mip_dir,'MIPDisplayDir':mip_display_dir,
                          'TissueSegDir':tissueseg_dir,'thDir':th_dir,'DispComp':dcomp}
            batches.append(batch_dict)
            
    if len(batches)<10:
        num_workers=len(batches)
        batches = [[x] for x in batches]
    else:
        num_workers=10
        batchsize = int(np.ceil(len(batches)/10))
        tmp=[]
        for i in range(num_workers):
            batch = batches[i*batchsize:(i+1)*batchsize]
            tmp.append(batch)
        batches=tmp
    
    pool = mp.Pool(num_workers)
    pool.map(get_class_map_MIPs,batches)
    pool.close()
         
    if not os.path.exists(os.path.join(save_dir,'MIP_pixel_summary.csv')):
        for sample in samples:
            if args.sample!='all_samples':
                if args.sample!=sample:
                    continue
            if not os.path.exists(os.path.join(save_dir,sample)):
                os.makedirs(os.path.join(save_dir,sample))
            areas = os.listdir(os.path.join(sam_dir,sample))
            for area in areas:
                if args.area!='all_areas':
                    if args.area!=area:
                        continue    
                if not os.path.exists(os.path.join(save_dir,sample,area)):
                    os.makedirs(os.path.join(save_dir,sample,area))
                px_dicts = []
                ims = os.listdir(os.path.join(mip_dir,sample,area))
                ims = [x for x in ims if (sample in x) and (area in x)]
                ims = [x for x in ims if x.endswith('.tif')]
                tims = os.listdir(tissueseg_dir)
                
                im = [x for x in ims if 'maxClass' in x]
                if len(im)==0:
                    print('****No MIP for',sample,area)
                    print('****Re-run',sample,area)
                    continue
                sample,area,_=im[0].split('_')
                orderfile = '_'.join([sample,area,'catOrder.pkl'])
                p1 = ['Bkgd']
                p2 = pkl.load(open(os.path.join(mip_dir,sample,area,orderfile),'rb'))
                p = p1+p2
                tim = [x for x in tims if (sample in x) and (area in x)]
                im2 = im[0].replace('Class','Score')
                timg = imread(os.path.join(tissueseg_dir,tim[0]))
                catimg = imread(os.path.join(mip_dir,sample,area,im[0]))
                tR,tC = np.shape(timg)
                R,C = np.shape(catimg)
                if (tR!=R) or (tC!=C):
                    timg = np.uint8(resize(timg,(R,C)))
                timg[timg>0]=1
                scoreimg = imread(os.path.join(mip_dir,sample,area,im2))
                scoreimg = scoreimg*timg
                catimg+=1
                catimg=catimg*timg
                cats,cts = np.unique(catimg,return_counts=True)
                for i,cat in enumerate(p):
                    pdict = {'Sample':sample,'Area':area,'CompName':sample+'_'+area,'Cat':cat}
                    if i in cats:
                        idx=list(cats).index(i)
                        pdict.update({'MaxCounts':cts[idx]})
                        pdict.update({'MeanScoreWhereMax':np.mean(scoreimg[catimg==i])})
                    else:  
                        pdict.update({'MaxCounts':0})
                        pdict.update({'MeanScoreWhereMax':0})
                    px_dicts.append(pdict)
                df = pd.DataFrame(px_dicts)
                df.to_csv(os.path.join(save_dir,'_'.join([sample,area,'MIP_pixel_summary.csv'])))
        
        csvs = os.listdir(save_dir)
        csvs = [x for x in csvs if x.endswith('.csv')]
        csvs = [x for x in csvs if len(x.split('_'))==5]
        if len(csvs)>1:
            if os.path.exists(os.path.join(save_dir,'MIP_pixel_summary.csv')):
                df = pd.read_csv(os.path.join(save_dir,'MIP_pixel_summary.csv'))
            else:
                df = pd.DataFrame()
            for csv in csvs:
                sdf = pd.read_csv(os.path.join(save_dir,csv))
                df = pd.concat([df,sdf],ignore_index=False)
                os.remove(os.path.join(save_dir,csv))
            df.to_csv(os.path.join(save_dir,'MIP_pixel_summary.csv'))
    else:
        df = pd.read_csv(os.path.join(save_dir,'MIP_pixel_summary.csv'))

    df2 = df[df['Cat']!='Bkgd']
    catcolors = np.unique(df2['Cat'])
    catcolors = [x for x in catcolors if 'NK' not in x]
    colors = [rgb_to_hex(colordict[x][0],colordict[x][1],colordict[x][2]) for x in catcolors]
    sn.set_palette(sn.color_palette(colors))
    dispnames = [dispname_dict[x] for x in catcolors]
    df2['MeanScoreWhereMax']=df2['MeanScoreWhereMax']/255
    for comp in np.unique(df2['CompName']):
        sample,area=comp.split('_')
        df3 = df2[df2['CompName']==comp]

        df3['Class Prevalence']=df3['MaxCounts']/df3['MaxCounts'].sum()
        sn.scatterplot(data=df3,x='Class Prevalence',y='MeanScoreWhereMax',
                   hue='Cat',hue_order=catcolors,sizes=[250]*len(catcolors))
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel('Pixel Class Prevalence',fontsize=8)
        plt.ylabel('Mean Class Score',fontsize=8)
        plt.legend(loc=2,bbox_to_anchor=(1.05,1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,sample,area,comp+'_MaxClass-by-MaxScore.png'),dpi=600)
        plt.close()
        
        plt.figure(figsize=(4,1.75))
        sn.barplot(data=df3,x='Cat',y='Class Prevalence',
                   hue='Cat',dodge=False,hue_order=catcolors,order=catcolors)
        plt.xticks(ticks=np.arange(0,len(catcolors)),labels=[])
        plt.yscale('log')
        plt.ylabel('Pixel Class Prevelance',fontsize=8)
        plt.xlabel('')
        plt.yticks(fontsize=6)
        plt.legend([],[],frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,sample,area,comp+'_Class-by-Max_log.png'),dpi=600)
        plt.close()

        plt.figure(figsize=(4,1.75))
        sn.barplot(data=df3,x='Cat',y='Class Prevalence',
                   hue='Cat',dodge=False,hue_order=catcolors,order=catcolors)
        plt.xticks(ticks=np.arange(0,len(catcolors)),labels=[])
        plt.ylabel('Pixel Class Prevelance',fontsize=8)
        plt.xlabel('')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend([],[],frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,sample,area,comp+'_Class-by-Max.png'),dpi=600)
        plt.close()

        plt.figure(figsize=(4,2.5))
        sn.barplot(data=df3,x='Cat',y='MeanScoreWhereMax',
                   hue='Cat',dodge=False,hue_order=catcolors,order=catcolors)
        plt.xticks(ticks=np.arange(0,len(catcolors)),labels=dispnames,rotation=90,fontsize=6)
        plt.ylabel('Mean Class Score',fontsize=8)
        plt.xlabel('')
        plt.ylim([0,1])
        plt.yticks(fontsize=6)
        plt.legend([],[],frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,sample,area,comp+'_Class-by-MaxScore.png'),dpi=600)
        plt.close()   
    
if __name__=='__main__':
    main()
