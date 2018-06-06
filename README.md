# Designing Random Graph Models Using Variational Autoencoders With Applications to Chemical Design
Paper : https://arxiv.org/pdf/1802.05283.pdf

## Required packages
-`tensorflow 1.4.1`

-`rdkit >= 2016.03.4`

-`networkx 2.0`

## Command to learn model

`python main.py --num_epochs 10 --learning_rate 0.0001 --log_every 5 --z_dim <z> --random_walk <k> --edges <e> --nodes <n> --graph_file <graph> --z_dir <zspace> --sample_file <sampledir> --out_dir <outputdir> >  log.out`

## Command to sample graph

`python sample.py --num_epochs 10 --learning_rate 0.0001 --log_every 5 --z_dim <z> --random_walk <k> --edges <e> --nodes <n> --graph_file <graph> --z_dir <zspace> --sample_file <sample> --out_dir <outputfile> > log.out`


## For real data please checkout the branch node_label
