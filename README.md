# Command for generating erdos renyi graph

python generate_erdos_renyi.py --n 10 --p 0.5 --m 2 --k 3

# Command for learning

python main_new.py --num_epochs=100 --learning_rate=0.001 --log_every 100 --graph_file <input file> --out_dir <output to store> --sample_file <sample> --z_dir <Zdir> --random_walk 5 --z_dim 7 --nodes <nodes to be sampled> --edges <edges to be sampled> --mask_weight True

# Command for sampling
python sample.py --num_epochs=100 --learning_rate=0.001 --log_every 100 --graph_file <input file> --out_dir <output to store> --sample_file <sample> --z_dir <Zdir> --random_walk 5 --z_dim 7 --nodes <nodes to be sampled> --edges <edges to be sampled> --mask_weight True
