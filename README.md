# Pseudospectral Angle Mapping (pSAM)

pSAM will calculate the similarity of every pixel in a high-plex image to a pre-defined set of reference 'pseudospectra'.

Within your dataset folder, you need to create a csv file that has all desired references. The first column of this file should be 'Marker', and contain all markers used to generate your references. All other columns will be the names for each reference pseudospectrum. For each marker, define an expected value ranging from 0-1 for each marker in your dataset (or each marker selected for the desired classifications).

## Scripts:

### spectral_angle_mapping.py: 
computes the cosine similarity to all references in 'reference_spectra.csv' for all pixels in the specified image(s).

Inputs: --dataset --sample --area --rootdir

Requires: reference_spectra.csv and contents of 'Normalized_composites' folder

Outputs/Creates: spectral_angle_mapping/class_maps folder

### find_pSAM_thresholds.py: 
computes the best threshold to use for low information-content experiments (<~10 channels and/or classes)

Inputs: --dataset --sample --area --rootdir

Requires: contents of spectral angle mapping/class_maps folder

Outputs/Creates: spectral_angle_mapping/map_thresholding folder

### pixel_class_analysis.py: 
makes maximum intensity projections of all class maps for the specified image. One MIP is encoded by class, while the other is a true MIP. Also generates visualizations of these MIPs and some basic plots for pixel class prevalence

Inputs: --dataset --sample --area --rootdir --pxsz (pixel size in microns)

Requires: full tissue segmentation stored in 'tissue_composite_masks' and output of spectral_angle_mapping.py

Outputs/Creates: SAM_analysis folder

### cellpose_wholeslide.py: 
gets nucleus segmentations for whole slide images; manages large images and cell counts by splitting files

Inputs: --dataset --sample --area --rootdir --pxsz (pixel size in microns)

Requires: full tissue segmentation stored in 'tissue_composite_masks' and DAPI images in 'Corrected_DAPI_composites'

Outputs/Creates: nucleus_segmentations_WS folder

### dilate_nuclear_segmentations-wholeSlide.py: 
dilates nucleus segmentations for whole slide images

Inputs: --dataset --sample --area --rootdir --pxsz (pixel size in microns) --pad (how much to dilate in microns) 

Requires: contents of nucleus_segmentations_WS folder

Outputs/Creates: wholeCell_segmentations_WS folder

### match_SAM_to_nuc-WS_eff.py: 
grabs cell and nucleus segmentations to predict cell classes based on pSAM class maps

Inputs: --dataset --sample --area --rootdir --pxsz (pixel size in microns)

Requires: nucleus and whole cell instance segmentations of full image/slide; full tissue segmentation stored in 'tissue_composite_masks_instance' and output of spectral_angle_mapping.py

Outputs/Creates: classify_by_dilation folder

### SAM_classify_from_score.py: 
takes pSAM scores per cell and gives class based on membrane comparment, nucleus compartment, and combined (final) class

Inputs: --dataset --sample --area --rootdir --pxsz (pixel size in microns)

Requires: SAM-score csv files from match_SAM... .py script

Outputs/Creates: 'FinalCats' csv files

### get_cell_MPI_NucMem_csv_eff.py: 
calculates cell MPI for all cells in segmentations; includes MPI for membrane and nucleus compartments and weighted MPI using blurred masks for whole cell (wc), membrane (mem), and nucleus (nuc) compartments

Inputs: --dataset --sample --area --rootdir

Requires: contents of Normalized_composites folder, whole cell segmentations stored at full-slide/full-image in wholeCell_segmentations_WS

Outputs/Creates: cell MPI csv in classify_by_dilation folder

### cellMPI_merge_sections.py: 
combines features from cells in large samples back into one file and adjusts CellIDs to reflect a single section of tissue; unique cellIDs per section are now under 'AdjCellID'

Inputs: --dataset --sample --area --rootdir

Requires: cell MPI csv 

Outputs/Creates: combined feature file in classify_by_dilation

### SAM_analysis_plots-general.py: 
combines cell feature csv and cell SAM score csv; generates prevalence plots and z-score MPI validation plot

Inputs: --dataset --sample --area --rootdir

Requires: combined feature file from merge script and 

Outputs/Creates: filtered csv file in classify_by_dilation/with-MPI; plots in classify_by_dilation/with-MPI/Prevalence_plots_filtered

### SAM_composite_display-WS.py: 
displays cell classes using whole cell segmentations in full-composite/full-image space

Inputs: --dataset --sample --area --rootdir

Requires: filtered csv from analysis script

Outputs/Creates: full-size and downsampled RGB visualizations of cell classes in 'Composite_predictions-withSAMclass'

### batch_pSAM.sh: 
runs all of the above scripts in order

Inputs: dataset sample area

Requires: reference_spectra.csv, contents of 'Normalized_composites' folder, masks in 'tissue_composite_masks_instance' folder

Outputs/Creats: see scripts above

## Database structure
The database should have the following structure:
> Dataset
>> reference_spectra.csv # pseudospectra refs
>> Corrected_DAPI_composites # DAPI images
>>> DAPIImage0
>>> ...
>>> DAPIImageX
>>> 
>> Normalized_composites #folder with all other image channels
>>> Sample1
>>>> Area1
>>>>> ImageChannel0
>>>>> ...
>>>>> ImageChannelN
>>>>> 
>> tissue_composite_masks_instance # tissue section or TMA core instance segmentations
>>> DAPIImage1
>>> ...
>>> DAPIImageX

## File naming
Image filenames should follow the convention:
target or marker _ wavelength or fluorophore _ cycleNumber _ dataset _ sampleID _ ROIid .tif

For example:
DAPI_UV_1_BC_120623S1_Area1.tif

CD4_AF647_9_BC_120623S1_Area1.tif

