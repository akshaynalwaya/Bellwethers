#! /bin/bash
#BSUB -W 6000
#BSUB -n 4
#BSUB -o ./out.%J
#BSUB -e ./err.%J

bsub -W 1000 -n 12 -q long -o ./out/out.1 -e ./err/err.1 /share/rkrish11/miniconda/bin/python2.7 par_exec.py
