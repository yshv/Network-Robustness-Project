#cd ~
#source ray_cluster.sh
#restart_cluster
conda activate dgl
#python Code/test/networktoolbox/scripts/Simulations/simulator_1.py & # Large scale dwc-select prufer sequence
#python Code/test/networktoolbox/scripts/Simulations/simulator_2.py & # Large scale dwc-select ER
#python Code/test/networktoolbox/scripts/Simulations/simulator_3.py & # Large scale dwc-select BA
#python Code/test/networktoolbox/scripts/Simulations/simulator_4.py & # Large scale random graphs BA, ER, SNR-BA, prufer-sequence - "ptd-random"
#python Code/test/networktoolbox/scripts/Simulations/simulator_5.py ; # Small scale random and dwc-select graphs for prufer-sequence
#fg
#wait
echo "stage 1 complete"
#python MPNN.py &&
python Simulations/simulator_6.py -m 300 # throughput calculations
echo "stage 2 complete"

#python Code/test/networktoolbox/scripts/prufer_sequence_topology_generation.py
#python /home/uceeatz/Code/test/networktoolbox/scripts/prufer\ sequence\ GA.py
#python /home/uceeatz/Code/test/networktoolbox/scripts/job_wait.py -p 'ray::GA_run()'
#python /home/uceeatz/Code/test/networktoolbox/scripts/throughput_calculation.py