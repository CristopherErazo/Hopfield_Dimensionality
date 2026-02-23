#!/bin/bash

echo 'Hola' 

Ns=(4096 8192 16384 32768)
Ts=($(seq 1.0 0.02 1.40))

# Loop over all combinations
for N in "${Ns[@]}"; do
    for T in "${Ts[@]}"; do
        JOB_NAME="N${N}"
	filename="./data/fss/N_samples10000_alpha0.04_burnin1500_h0/summary_N${N}_T${T}.txt"
	# echo $N $T
	echo $(wc $filename)
done
done
