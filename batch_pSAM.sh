export DATASET=$1
export SAMPLE=${2:-'all_samples'}
export AREA=${3:-'all_areas'}

echo ""
echo "Running pSAM. You can check output while running and stop if progress is not acceptable using ctrl+c"

echo ""
echo "Calculating class maps..."
python3 spectral_angle_mapping.py --dataset $DATASET --sample $SAMPLE --area $AREA 
echo ""
echo "Class maps calculated."

echo ""
echo "Running pixel SAM analysis..."
python3 pixel_class_analysis.py --dataset $DATASET --sample $SAMPLE --area $AREA
echo ""
echo "MIPs and plots saved."

echo ""
echo "Collecting cell segmentations..."
source activate cellpose
cd ../cell_instance_segmentation/CellPose
python3 cellpose_wholeSlide.py --dataset $DATASET --sample $SAMPLE --area $AREA
cd ../nuclear_dilation
conda deactivate
python3 dilate_nuclear_segmentations-wholeSlide.py --dataset $DATASET --sample $SAMPLE --area $AREA
cd ../../spectral_angle_mapping
echo ""
echo "Cell segmentations done."


echo ""
echo "Assigning pSAM class to segmented nuclei..."
python3 match_SAM_to_nuc-WS-fast.py --dataset $DATASET --sample $SAMPLE --area $AREA
echo ""
echo "Classes assigned."

echo ""
echo "Calculating MPIs for each classified cell..."
python3 get_cell_MPI_csv.py --dataset $DATASET --sample $SAMPLE --area $AREA
python3 merge_class_and_MPI_csvs.py --dataset $DATASET --sample $SAMPLE --area $AREA
echo ""
echo "MPI csv generated."

echo ""
echo "Plotting class info..."
python3 SAM_analysis_plots-general.py --dataset $DATASET --sample $SAMPLE --area $AREA
echo ""
echo "Class info plotted."

echo ""
echo "Generating cell class visualizations..."
python3 SAM_class_composite_display-WS.py --dataset $DATASET --sample $SAMPLE --area $AREA
echo ""
echo "Composite displays saved."

echo ""
echo "Done with pSAM. Check results."
