#!/bin/bash

#cd ~
#source ray_cluster.sh
#restart_cluster
#conda info --envsc
#source /home/uceeatz/miniconda3/etc/profile.d/conda.sh
conda activate dgl &&
python /home/uceeatz/Code/networktoolbox/scripts/PTD-sim/htd_distance_large.py &&
python /home/uceeatz/Code/networktoolbox/scripts/PTD-sim/prufer_ga.py &&
python /home/uceeatz/Code/networktoolbox/scripts/PTD-sim/prufer_random_dwc_select_distance_large.py &&
python /home/uceeatz/Code/networktoolbox/scripts/PTD-sim/topology_vector_ga.py

