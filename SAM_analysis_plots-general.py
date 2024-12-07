#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:49:49 2023

@author: durkeems
"""
import os
import argparse
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu,threshold_multiotsu
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
  return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def filter_cells(df):
    cats = [x for x in df.columns if 'Int-mean' not in x]
    cats = [x for x in cats if 'Centroid' not in x]
    cats = [x for x in cats if x not in ['Sample','Area','CellID','CompName']]
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
    memcatname = [x if memmax[i]>0 else 'unclassified' for i,x in enumerate(memcatname)] #0
    nuccatname = [x if nucmax[i]>0 else 'Nucleus' for i,x in enumerate(nuccatname)] #0
    df['MemCat']=memcat
    df['NucCat']=nuccat
    df['MemCatName']=memcatname
    df['NucCatName']=nuccatname
    df['MemCatScore']=memmax
    df['NucCatScore']=nucmax

    return df    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/',help='')
    parser.add_argument('--dataset',type=str,default='PSC_CRC',help='')
    parser.add_argument('--readdir',type=str,default='classify_by_dilation/with-MPI',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    
    args,unparsed = parser.parse_known_args()
    
    rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
    sdir = os.path.join(rdir,'Prevalence_Plots_filtered')
    
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if (args.sample != 'all_samples') and (args.area !='all_areas'):
        csv_names = os.listdir(os.path.join(rdir,args.sample,args.area))
        csv_names = ['/'.join([args.sample,args.area,x]) for x in csv_names]
        csv_names = [x for x in csv_names if 'cellMPI' not in x]
    else:
        csv_names = ['combined_SAM-scores_with-MPI.csv']

    for csv_name in csv_names:
        csvname=csv_name.split('_SAM')[0].split('/')[-1]
    
        df = pd.read_csv(os.path.join(rdir,csv_name))
        df = filter_cells(df)
        df.to_csv(os.path.join(rdir,csvname+'_SAM_scores_filtered.csv'))
        
        catnames = df['MemCatName']
        catnames = np.unique(catnames)

        memcolor_dict = {}
    
        if args.dataset=='PSC_CRC':
            #PSC panel
            memcolor_dict={'APC':[155,5,255],'ActivatedCD8T':[10,242,14],'B':[28,25,227],
                       'CytotoxicT':[6,140,8],'HelperT':[237,14,29],'IL17-HelperT':[166,3,3],
                       'IL17CD45':[247,7,235],'IgG-Plasma':[9,237,188],'OtherT':[246,250,5],
                       'OtherCD45':[123,7,116],'Plasma':[10,166,132],'Plasmablast':[45,148,196],
                       'unclassified':[168,167,165], 'structure':[133,113,60]}
            mem_names = ['APC','CD8T-Act','B','CD8T','CD4T','IL17+ CD4T','IL17+ Other Immune','IgG+ Plasma',
                        'Other T','Other Immune','Plasma','Plasmablast','Unclassified','Structural']
            memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolor_dict.keys()))}
        else:
            #kidney panel
            memcolor_dict={'B':[28,25,227],'B-TRM':[74,131,237],'BowmansCapsule':[117,74,8],'CD4CD8T':[246,250,5],
                       'CD4T':[237,14,29],'CD4T-ICOS':[166,3,3],'CD4T-ICOSPD1':[245,69,69],'CD4T-PD1':[150,47,47],'CD4T-TRM':[219,116,116],
                       'CD8T':[10,242,14],'CD8T-Ex':[6,140,8],'CD8T-GZB':[69,153,70],'CD8T-TRM':[98,166,70],
                       'DistalTubule':[133,113,60],'Endothelial cell':[186,174,141],'GDT':[247,7,235],'HealthyTubule':[176,136,4],'Glom':[148,123,43],
                       'M2':[71,204,180],'Macrophage-CD14+':[61,245,242],'Macrophage-CD16+':[3,171,168],'Monocyte-GZB':[157,227,245],
                       'Monocyte-HLAII+':[0,167,245],'Monocyte-HLAII-':[45,148,196],'NK':[236,245,157],'NKT':[245,210,157],'Neutrophils':[237,120,9],
                       'Plasma':[9,237,188],'Plasma-Act':[10,166,132],'Plasmablast':[59,148,106],'ProximalTubule':[150,142,117],
                       'cDC1':[155,5,255],'cDC2':[100,32,145],'pDC':[191,121,237],'unclassified':[168,167,165]}
            
            mem_names = ['B','TRM B','Stressed Tubule','CD4CD8T','CD4T','ICOS+ CD4T','ICOS+PD1+ CD4T','PD1+ CD4T',
                         'TRM CD4T','CD8T','Exhausted CD8T','GZM+ CD8T','TRM CD8T','Distal Tubule','Endothelial','\u03B3\u03B4 T',
                         'Healthy Tubule','Other Glomerulus','CD163+ Macrophage','Other Macrophage','CD16+ Macrophage','GZB+ Monocyte','HLAII+ Monocyte',
                         'HLAII- Monocyte','NK','NKT','Neutrophil','Plasma','SLAMF7+ Plasma','Plasmablast','Proximal Tubule',
                         'cDC1','cDC2','pDC','Unclassified']
            memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolor_dict.keys()))}
    
        catnames = df['MemCatName']
        catnames = np.unique(catnames)
        catnames.sort()

        cols = df.columns
        mpicols = [x for x in cols if 'Int-mean' in x]
        mpidf = df[mpicols]
        cols = [x for x in cols if 'Int-mean' not in x]
        
        df = df[cols]
        cats = np.unique(df['MemCatName'])
        samples = np.unique(df['CompName'])
        
        #for anonymizing final figs
        anonsamples = list(samples)
        usamples = np.unique([x.split('_')[0] for x in anonsamples])
        for idx,sname in enumerate(usamples):
            newname = 'S'+str(idx+1)
            anonsamples = [x.replace(sname,newname) for x in anonsamples]
        anonsamples = [x.replace('Area','A') for x in anonsamples]
        cellcts = {}
        for sample in samples:
            sdf = df[df['CompName']==sample]
            cells,_=sdf.shape
            cellcts.update({sample:cells})
            
        if args.dataset=='PSC_CRC':
            ordered_cats=[x for x in memcolor_dict.keys()]
        else:
            ordered_cats = ['BowmansCapsule','DistalTubule','Endothelial cell',
                            'Glom','HealthyTubule','ProximalTubule',
                            'B','B-TRM','CD4T','CD4T-ICOS',
                            'CD4T-ICOSPD1','CD4T-PD1','CD4T-TRM','CD8T','CD8T-Ex',
                            'CD8T-GZB','CD8T-TRM','NKT','GDT','CD4CD8T','M2','Macrophage-CD14+',
                            'Macrophage-CD16+','Monocyte-GZB','Monocyte-HLAII+',
                            'Monocyte-HLAII-','NK','Neutrophils','Plasma',
                            'Plasma-Act','Plasmablast','cDC1','cDC2','pDC','unclassified']
        cts = df.groupby('CompName')['MemCat'].count()
        cell_fractions=pd.DataFrame()
        for sample in samples:
            subdf = df[df['CompName']==sample]
            cat_props={}
            for cat in ordered_cats:
                pdf = subdf[subdf['MemCatName']==cat]
                props = pdf['MemCatName'].count()/subdf.shape[0]
                cat_props.update({cat:[props]})
            catdf = pd.DataFrame.from_dict(cat_props)
            cell_fractions=pd.concat([cell_fractions,catdf],ignore_index=True)
        ordered_colors = [memcolor_dict[x] for x in ordered_cats if x in cell_fractions.columns]
        print(cell_fractions)
        print(cell_fractions[ordered_cats].sum(axis=0))
        cell_fractions['Bx-Image']=anonsamples
        cell_fractions['CompName']=samples
        cell_fractions.plot(kind='bar',x='Bx-Image',stacked=True,width=0.9,
                            color=[rgb_to_hex(x) for x in ordered_colors],rot=0)
        #plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
        plt.legend('',frameon=False)
        plt.ylabel('Proportion of Cells',fontsize=12)
        plt.xlabel('')
        plt.xticks(fontsize=11,rotation=0)
        plt.yticks(ticks=[0,0.5,1.0],fontsize=11)
        plt.title('Membrane Type Prevalence')
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'CellTypePrevalence_'+csvname+'.png'),dpi=600)
        #plt.show()
        plt.close()
       
        samples = np.unique(df['Sample'])
        
        #for anonymizing final figs
        anonsamples = list(samples)
        usamples = np.unique([x.split('_')[0] for x in anonsamples])
        for idx,sname in enumerate(usamples):
            newname = 'S'+str(idx+1)
            anonsamples = [x.replace(sname,newname) for x in anonsamples]
        cellcts = {}
        for sample in samples:
            sdf = df[df['Sample']==sample]
            cells,_=sdf.shape
            cellcts.update({sample:cells})
            
        cts = df.groupby('Sample')['MemCat'].count()
        
        cell_fractions=pd.DataFrame()
        for sample in samples:
            subdf = df[df['Sample']==sample]
            cat_props={}
            for cat in cats:
                pdf = subdf[subdf['MemCatName']==cat]
                props = pdf['MemCatName'].count()/subdf.shape[0]
                cat_props.update({cat:[props]})
            catdf = pd.DataFrame.from_dict(cat_props)
            cell_fractions=pd.concat([cell_fractions,catdf],ignore_index=True)
        if args.dataset=='PSC_CRC':
            ordered_cats=[x for x in memcolor_dict.keys()]
        else:
            ordered_cats = ['BowmansCapsule','DistalTubule','Endothelial cell',
                            'Glom','HealthyTubule','ProximalTubule',
                            'B','B-TRM','CD4T','CD4T-ICOS',
                            'CD4T-ICOSPD1','CD4T-PD1','CD4T-TRM','CD8T','CD8T-Ex',
                            'CD8T-GZB','CD8T-TRM','NKT','GDT','CD4CD8T','M2','Macrophage-CD14+',
                            'Macrophage-CD16+','Monocyte-GZB','Monocyte-HLAII+',
                            'Monocyte-HLAII-','NK','Neutrophils','Plasma',
                            'Plasma-Act','Plasmablast','cDC1','cDC2','pDC','unclassified']
        ordered_colors = [memcolor_dict[x] for x in ordered_cats if x in cell_fractions.columns]
        cell_fractions['Bx-Image']=anonsamples
        cell_fractions['Sample']=samples
        cell_fractions.plot(kind='bar',x='Bx-Image',stacked=True,width=0.9,
                            color=[rgb_to_hex(x) for x in ordered_colors])
        #plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
        plt.legend('',frameon=False)
        plt.ylabel('Proportion of Cells',fontsize=14)
        plt.xticks(fontsize=12,rotation=0)
        plt.yticks(ticks=[0,0.5,1.0],fontsize=12)
        plt.xlabel('')
        plt.title('Membrane Class Prevalence',fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'SampleCellTypePrevalence_'+csvname+'.png'),dpi=600)
        #plt.show()
        plt.close()
        
        ########################       
        catnames = df['NucCatName']
        catnames = np.unique(catnames)
        catnames.sort()

        nuccolor_dict = {}
    
        #selected nucleus colors
        if args.dataset=='PSC_CRC':
            #PSC panel
            nuccolor_dict={'Nucleus':[200,200,200],'RegT-Nuc':[135,16,88],'pSTAT3-Nuc':[252,152,3]}
            nuc_names = ['Nucleus','Foxp3+ Nucleus','pSTAT3+ Nucleus']
            nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolor_dict.keys()))}
        else:
            #kidney panel
            nuccolor_dict={'Nucleus':[200,200,200],'Nucleus-FoxP3+':[135,16,88],
                           'Nucleus-Ki67+':[2,94,37],'Nucleus-Tubule':[179,144,4]}
            nuc_names = ['Nucleus','Foxp3+ Nucleus','Ki67+ Nucleus','Tubule Nucleus']
            nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolor_dict.keys()))}
    
        cats = np.unique(df['NucCatName'])

        samples = np.unique(df['CompName'])
        cellcts = {}
        for sample in samples:
            sdf = df[df['CompName']==sample]
            cells,_=sdf.shape        
            cellcts.update({sample:cells})
            
        cts = df.groupby('CompName')['NucCat'].count()
        
        anonsamples = list(samples)
        usamples = np.unique([x.split('_')[0] for x in anonsamples])
        for idx,sname in enumerate(usamples):
            newname = 'S'+str(idx+1)
            anonsamples = [x.replace(sname,newname) for x in anonsamples]
        anonsamples = [x.replace('Area','A') for x in anonsamples]
        
        cell_fractions=pd.DataFrame()
        for sample in samples:
            subdf = df[df['CompName']==sample]
            cat_props={}
            for cat in cats:
                pdf = subdf[subdf['NucCatName']==cat]
                props = pdf['NucCatName'].count()/cts[sample]
                cat_props.update({cat:[props]})
            catdf = pd.DataFrame.from_dict(cat_props)
            cell_fractions=pd.concat([cell_fractions,catdf],ignore_index=True)

        ordered_colors = [nuccolor_dict[x] for x in nuccolor_dict.keys() if x in cell_fractions.columns]
        cell_fractions['Bx-Image']=anonsamples
        cell_fractions['CompName']=samples
        cell_fractions.plot(kind='bar',x='Bx-Image',stacked=True,width=0.9,
                            color=[rgb_to_hex(x) for x in ordered_colors])
        #plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
        plt.legend('',frameon=False)
        plt.ylabel('Proportion of Cells',fontsize=12)
        plt.xlabel('')
        plt.xticks(fontsize=11,rotation=0)
        plt.yticks(ticks=[0,0.5,1.0],fontsize=11)
        plt.title('Nucleus Class Prevalence')
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'NucleusTypePrevalence_'+csvname+'.png'),dpi=600)
        #plt.show()
        plt.close()
        
        samples = np.unique(df['Sample'])
        cellcts = {}
        for sample in samples:
            sdf = df[df['Sample']==sample]
            cells,_=sdf.shape        
            cellcts.update({sample:cells})
            
        cts = df.groupby('Sample')['NucCat'].count()
        
        anonsamples = list(samples)
        usamples = np.unique([x.split('_')[0] for x in anonsamples])
        for idx,sname in enumerate(usamples):
            newname = 'S'+str(idx+1)
            anonsamples = [x.replace(sname,newname) for x in anonsamples]
        
        cell_fractions=pd.DataFrame()
        for sample in samples:
            subdf = df[df['Sample']==sample]
            cat_props={}
            for cat in cats:
                pdf = subdf[subdf['NucCatName']==cat]
                props = pdf['NucCatName'].count()/cts[sample]
                cat_props.update({cat:[props]})
            catdf = pd.DataFrame.from_dict(cat_props)
            cell_fractions=pd.concat([cell_fractions,catdf],ignore_index=True)
        
        ordered_colors = [nuccolor_dict[x] for x in nuccolor_dict.keys() if x in cell_fractions.columns]
        cell_fractions['Bx-Image']=anonsamples
        cell_fractions['Sample']=samples
        cell_fractions.plot(kind='bar',x='Bx-Image',stacked=True,width=0.9,
                            color=[rgb_to_hex(x) for x in ordered_colors])
        #plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
        plt.legend('',frameon=False)
        plt.ylabel('Proportion of Cells',fontsize=14)
        plt.xticks(fontsize=12,rotation=0)
        plt.yticks(ticks=[0,0.5,1.0],fontsize=12)
        plt.xlabel('')
        plt.title('Nucleus Class Prevalence',fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'SampleNucleusTypePrevalence_'+csvname+'.png'),dpi=600)
        #plt.show()
        plt.close()
        '''
        #class matching plots
        N = list(nuccolor_dict.keys())
        for n in N:
            if n not in np.unique(df['NucCatName']):
                continue
            ndf = df[df['NucCatName']==n]
            mcats = ndf.groupby('MemCatName')['MemCat'].count()
            mcolors = [memcolor_dict[x] for x in mcats.index]
            mcolors = [[x[0]/255,x[1]/255,x[2]/255] for x in mcolors]
            plt.pie(mcats,colors=mcolors)
            plt.title(nucname_dict[n]+'\nN= '+str(ndf.shape[0]),fontsize=9)
            plt.savefig(os.path.join(sdir,n+'_cellClassDist.png'),dpi=600)
            #plt.show()
        
        M = list(memcolor_dict.keys())
        for m in M:
            if m not in np.unique(df['MemCatName']):
                continue
            mdf = df[df['MemCatName']==m]
            ncats = mdf.groupby('NucCatName')['NucCat'].count()
            ncolors = [nuccolor_dict[x] for x in ncats.index]
            ncolors = [[x[0]/255,x[1]/255,x[2]/255] for x in ncolors]
            plt.pie(ncats,colors=ncolors)
            plt.title(memname_dict[m]+'\nN= '+str(mdf.shape[0]),fontsize=9)
            plt.savefig(os.path.join(sdir,m+'_cellClassDist.png'),dpi=600)
            #plt.show()
        '''
        
        nuccats = np.unique(df['NucCatName'])
        memcats = np.unique(df['MemCatName'])
        cats = list(memcats)+list(nuccats)
        
        mpicols = [x for x in mpicols if ('TAF' not in x) and ('DIC' not in x)]
        mpicols.sort()
        for col in mpicols:
            df[col]=mpidf[col].dropna()
            df[col+'_Z'] = (mpidf[col]-mpidf[col].mean())/mpidf[col].std()
        mpi_cols = [x for x in df.columns if 'Int-mean' in x]
        mpi_cols.sort()

        mu_array = np.zeros([len(memcats)+len(nuccats),len(mpicols)])
        z_array = np.zeros([len(memcats)+len(nuccats),len(mpicols)])
        
        for i,cat in enumerate(memcats):
            mpi_df = df[df['MemCatName']==cat][mpi_cols]
            for j,col in enumerate(mpicols):
                mu = mpi_df[col].dropna().mean()
                z = mpi_df[col+'_Z'].dropna().mean()
                mu_array[i,j]=mu
                z_array[i,j]=z
        for ii,cat in enumerate(nuccats):
            idx = ii+i+1
            mpi_df = df[df['NucCatName']==cat][mpi_cols]
            for j,col in enumerate(mpicols):
                mu = np.mean(mpi_df[col])
                z = mpi_df[col+'_Z'].mean()
                mu_array[idx,j]=mu
                z_array[idx,j]=z
 
        markers = [x.split('_Int')[0] for x in mpicols]
        plt.figure(figsize=(4,4),dpi=600)
        plt.imshow(mu_array,cmap='jet')
        plt.yticks(ticks=np.arange(0,len(cats)),labels=cats,fontsize=5)
        plt.xticks(ticks=np.arange(0,len(mpicols)),labels=markers,rotation=90,fontsize=5)
        plt.colorbar()
        plt.xlabel('Marker',fontsize=8)
        plt.ylabel('Predicted Cell Class',fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'MeanMPI-perClass.png'),dpi=600)
        #plt.show()
        plt.close()
        cats = [memname_dict[x] for x in memcats] + [nucname_dict[x] for x in nuccats]
        
        if args.dataset=='PSC_CRC':
            cats = ['APC','GZB+ CD8T cell','B cell','CD8T cell','CD4T cell',
                    'IL17+ CD4T cell','IL17+ Other Immune cell','IgG+ Plasma cell',
                    'Other Immune cell','Other T cell','Plasma cell','Plasmablast',
                    'Structural cell','Unclassified cell','Generic Nucleus',
                    'Foxp3+ Nucleus','pSTAT3+ Nucleus']
        else:
            cats = ['B cell','TRM B cell','Stressed tubule cell','CD4CD8T cell','CD4T cell',
                    'ICOS+ CD4T cell','ICOS+PD1+ CD4T cell','PD1+ CD4T cell','TRM CD4T cell',
                    'CD8T cell','Exhausted CD8T cell','GZM+ CD8T cell','TRM CD8T cell',
                    'Distal tubule cell','Endothelial cell','\u03B3\u03B4 T cell','Glomerular cell',
                    'Healthy tubule cell','CD163+ Macrophage','Other Macrophage','CD16+ Macrophage',
                    'GZB+ Monocyte','HLAII+ Monocyte','HLAII- Monocyte','NK cell','NKT cell','Neutrophil',
                    'Plasma cell','SLAMF7+ Plasma cell','Plasmablast','Proximal tubule cell',
                    'cDC1','cDC2','pDC','Unclassified cell','Generic Nucleus','Foxp3+ Nucleus',
                    'Ki67+ Nucleus','Tubule Nucleus']
            
        markers = [x.replace('gamma','\u03B3') for x in markers]  
        markers = [x.replace('delta','\u03B4') for x in markers]
        import matplotlib as mpl
        mpl.rcParams['font.size']=8 
        plt.figure(figsize=(6.5,4.5),dpi=600)
        plt.imshow(z_array,vmin=-3,vmax=3,cmap='seismic')
        plt.yticks(ticks=np.arange(0,len(cats)),labels=cats,fontsize=8)
        plt.xticks(ticks=np.arange(0,len(mpicols)),labels=markers,fontsize=8,rotation=90)
        plt.colorbar()
        plt.xlabel('Marker',fontsize=8)
        plt.ylabel('Predicted Cell or Nucleus Class',fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(sdir,'MeanMPIzscore-perClass.png'),dpi=600)
        #plt.show()
        plt.close()
        
if __name__=="__main__":
    main()
