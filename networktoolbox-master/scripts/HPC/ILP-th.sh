#!/bin/bash -l

# Request ten minutes of wallclock time (format hours:minutes:seconds).
$ -l h_rt=10:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
$ -l mem=1G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
$ -l tmpfs=15G

# Set up the job array.  In this instance we have requested 10000 tasks
# numbered 1 to 10000.
$ -t 1-400

# Set the name of the job.
$ -N ILP-th

# setup ssh tunnel
ssh -L
conda activate networktoolkit
python run_ilp.py -t $SGE_TASK_ID