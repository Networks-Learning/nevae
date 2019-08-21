#!/bin/bash


nodes=( 30 31 32 33 34 35 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 )
#nodes=(30)
a=0

while [ $a -lt $1 ]
do
	for i in ${nodes[@]}
	do
		python main.py --num_epochs=10 --learning_rate=0.005 --log_every 100 --graph_file=data/n_$i --out_dir=output_with_mask/ --sample_file sample_with_mask/ --z_dir zspace_with_mask/ --random_walk 5 --z_dim 7 --nodes $i --mask_weight True >  output_with_mask/nohup.out
	done
	a=`expr $a + 1`
done
