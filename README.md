# Command for generating erdos renyi graph

python generate_erdos_renyi.py --n 10 --p 0.5 --m 2 --k 3

# Command for learning

python main.py --num_epochs=1 --learning_rate=0.001 --log_every 20 --graph_file=graph/ --out_dir=output/ --sample_file=sample/ --random_walk=3 --z_dir plot_z/
