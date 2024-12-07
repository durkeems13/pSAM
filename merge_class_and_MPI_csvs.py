import os
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='PSC_CRC',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    parser.add_argument('--readdir',type=str,default='classify_by_dilation',help='')
    parser.add_argument('--savedir',type=str,default='with-MPI',help='')

    args,unparsed = parser.parse_known_args()

    rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
    sdir = os.path.join(rdir,args.savedir)

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    samples = os.listdir(rdir)
    if 'combined_SAM-scores.csv' in samples:
        mpi_csv_path = os.path.join(rdir,'cellMPI.csv')
        mpidf=pd.read_csv(mpi_csv_path)
        catdf=pd.read_csv(os.path.join(rdir,'combined_SAM-scores.csv'))
        mdf=catdf.merge(mpidf,on=['CompName','CellID'])
        keepcols = [x for x in mdf.columns if not x.endswith('_y')]
        renamecols = [x for x in mdf.columns if x.endswith('_x')]
        newnames = [x.replace('_x','') for x in renamecols]
        mdf=mdf[keepcols]
        mdf[newnames]=mdf[renamecols]
        mdf=mdf.drop(columns=renamecols)
        mdf.to_csv(os.path.join(sdir,'combined_SAM-scores_with-MPI.csv'))
    else:
        samples = [x for x in samples if x!='with-MPI']
        samples = [x for x in samples if not x.endswith('.csv')]
        for sample in samples:
            if args.sample!='all_samples':
                if args.sample!=sample:
                    continue
            areas = os.listdir(os.path.join(rdir,sample))
            if not os.path.exists(os.path.join(sdir,sample)):
                os.makedirs(os.path.join(sdir,sample))
            for area in areas:
                if args.area!='all_areas':
                    if args.area!=area:
                        continue
                if not os.path.exists(os.path.join(sdir,sample,area)):
                    os.makedirs(os.path.join(sdir,sample,area))
                csvs = os.listdir(os.path.join(rdir,sample,area))
                csvs = [x for x in csvs if 'MPI' not in x]
                mpiname='_'.join([sample,area,'cellMPI.csv'])
                mpi_csv_path = os.path.join(sdir,sample,area,mpiname)
                mpidf = pd.read_csv(mpi_csv_path)
                print(csvs[0])
                print(mpidf.shape)
                catdf=pd.read_csv(os.path.join(rdir,sample,area,csvs[0]))
                print(catdf.shape)
                mdf=catdf.merge(mpidf,on=['CompName','CellID'])
                keepcols = [x for x in mdf.columns if not x.endswith('_y')]
                renamecols = [x for x in mdf.columns if x.endswith('_x')]
                newnames = [x.replace('_x','') for x in renamecols]
                mdf=mdf[keepcols]
                mdf[newnames]=mdf[renamecols]
                mdf=mdf.drop(columns=renamecols)
                mdf.to_csv(os.path.join(sdir,sample,area,sample+'_'+area+'_SAM-scores_with-MPI.csv'))

if __name__=="__main__":
    main()
