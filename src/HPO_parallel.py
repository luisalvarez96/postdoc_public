import argparse
from multiprocessing import process
import os
import sys
import subprocess
import time

def parse_args():
    parser = argparse.ArgumentParser(description='HPO for HGTDR')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--metric', type=str, default='Val_Loss', help='Metric to optimize during HPO, must be one of AUROC, AUPR, or Val Loss')
    parser.add_argument('--n_calls', type=int, default=50, help='Number of hyperparameter configurations to evaluate during HPO')

    parser.add_argument('--sampling_layers', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=314)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--node_type1', type=str, default='drug')
    parser.add_argument('--node_type2', type=str, default='disease')
    parser.add_argument('--rel', type=str, default='indication')
    parser.add_argument('--seeded_type', type=str, default='disease', help='Type of node to be seeded')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    OUTER_CVs_DIR = args.data_path
    outer_cvs_data = [d for d in os.listdir(OUTER_CVs_DIR) if d.startswith('CV') and os.path.isdir(os.path.join(OUTER_CVs_DIR, d))]

    os.makedirs(args.out_path, exist_ok=True)
    outer_cvs_out = [d for d in os.listdir(args.out_path) if d.startswith('CV') and os.path.isdir(os.path.join(args.out_path, d))]

    start = time.time()
    processes = []
    print(f"\nStarting HPO for {len(outer_cvs_data)} CV sets...")
    for dir in outer_cvs_data:
        OUTER_CV_DATA_PATH = os.path.join(OUTER_CVs_DIR, dir)
        out_path = os.path.join(args.out_path, dir)
        log_path = os.path.join(out_path, 'hpo_log.txt')

        os.makedirs(out_path, exist_ok=True)

        cmd = f"python HPO.py --data_path {OUTER_CV_DATA_PATH} " \
              f"--out_path {out_path} " \
              f"--metric {args.metric} " \
              f"--n_calls {args.n_calls} " \
              f"--sampling_layers {args.sampling_layers} " \
              f"--num_samples {args.num_samples} " \
              f"--batch_size {args.batch_size} " \
              f"--epochs {args.epochs} " \
              f"--node_type1 {args.node_type1} " \
              f"--node_type2 {args.node_type2} " \
              f"--rel {args.rel} " \
              f"--seeded_type {args.seeded_type}"
        
        with open(log_path, 'w') as log_file:
            p = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
            # sys.stdout = log_file
            # sys.stderr = log_file
            processes.append(p)
        print(f"Started subprocess {p.pid}: {cmd} > {log_path}")
    print()
    for p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Subprocess {p.pid} failed with return code {p.returncode}. Check the log file for details.")
        else:
            print(f"Subprocess {p.pid} completed successfully.")

    print(f"\nTotal run time: {(time.time() - start) / 60:.2f} minutes.")