import os
import sys
import subprocess
import torch
import argparse
import time
import shutil
import pandas as pd

from utils_sampling_processing import process_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_layers', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=164)
    parser.add_argument('--channels', nargs='+', type=int, default=[64])
    parser.add_argument('--out_channel', type=int, default=64)
    parser.add_argument('--att_heads', nargs='+', type=int, default=[8, 8, 8])
    parser.add_argument('--GNN_layers', type=int, default=3)
    parser.add_argument('--GNN_dropout', type=float, default=0.5)
    parser.add_argument('--MLP_dropout', type=float, default=0.2)
    # parser.add_argument('--ind_MSG_Pass', action='store_true', help='Use indications for message passing')
    # parser.add_argument('--no-ind_MSG_Pass', dest='ind_MSG_Pass', action='store_false', help='Do not use indications for message passing')
    # parser.set_defaults(ind_MSG_Pass=True)
    parser.add_argument('--seeded_ind_only', action='store_true', help='Use ONLY indications from SEEDED NODES for batches')
    parser.add_argument('--no-seeded_ind_only', dest='seeded_ind_only', action='store_false', help='Use indications from UNSEEDED NODES for message passing')
    parser.set_defaults(seeded_ind_only=False)
    parser.add_argument('--sup_split', type=float, default=.8, help='Percentage of indications used for supervision (as opposed to message passing) during training')
    parser.add_argument('--balanced', action='store_true', help='Balance contraindications with indications for each batch prediction')
    parser.add_argument('--not-balanced', dest='balanced', action='store_false')
    parser.set_defaults(balanced=False)
    parser.add_argument('--Residuals', action='store_true', help='Use residual connections')
    parser.add_argument('--no-Residuals', dest='Residuals', action='store_false', help='Do not use residual connections')
    parser.set_defaults(Residuals=True)

    parser.add_argument('--node_type1', type=str, default='drug')
    parser.add_argument('--node_type2', type=str, default='disease')
    parser.add_argument('--rel', type=str, default='indication')
    parser.add_argument('--seeded_type', type=str, default='disease')  # Type of node to be seeded

    parser.add_argument('--workers', type=int, default=1) # 8
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--out_path', type=str, default='../out') # /saved_model')
    parser.add_argument('--data_path', type=str, default='../data') # /hetero_graph.pt')
    parser.add_argument('--split', nargs='+', type=float, default=[.15, .10], help='List providing % for validation (in each CV set) and for test (across all CV sets)')
    # parser.add_argument('--CV_sets', type=int, default=5, help='Number of cross-validation sets')

    parser.add_argument('--global_neg_edges', action='store_true', help='Use global negative edges')
    parser.add_argument('--no-global_neg_edges', dest='global_neg_edges', action='store_false', help='Do not use global negative edges')
    parser.set_defaults(global_neg_edges=False)
    
    return parser.parse_args()

def about_data(data, config):
    print(f"\nLoaded HeteroData data with {len(data.node_types)} node types and {len(data.edge_types)} edge types.")

    print("Node types and their embeddings:")
    for node_type in data.node_types:
        print(data[node_type].x.shape, '\t', node_type)
    
    print(f"\nTotal number of edges: {data.num_edges}")
    for edge_type in data.edge_types:
        if edge_type[0] == 'disease': # or edge_type[2] == 'disease':
            if edge_type[1] == config['rel'] and 'edge_label' in data[edge_type]:
                print('LABELED INDICATION EDGES in data:')
                labels = data[edge_type].edge_label
                print('\tIndications = ', (labels == 1).sum().item())
                print('\tContraindications = ', (labels == 2).sum().item())
                if 0 in labels:
                    print('\tNeg edges = ', (labels == 0).sum().item())

            else:
                print(f"Number of {edge_type[1]}: {data[edge_type].num_edges}")
    print(f"Number of drug_protein:\t{data[('drug', 'drug_protein', 'gene/protein')].num_edges}\n")

def collect_metrics(base_dir = "../out"):
    """
    Reads metric values from the final lines of 'out.txt' files in subdirectories
    and compiles them into a pandas DataFrame.
    """
    all_metrics = []

    cv_out = [d for d in os.listdir(base_dir) if d.startswith('CV') and os.path.isdir(os.path.join(base_dir, d))]

    # Iterate through all subdirectories in the base directory
    for dir in cv_out:
        cv_dir_path = os.path.join(base_dir, dir)

        # Check if the path is a directory and contains the output file
        # if os.path.isdir(cv_dir_path):
        log_file_path = os.path.join(cv_dir_path, 'out.txt')
        if os.path.exists(log_file_path):

            # Extract the fold name from the directory name
            # fold_name = dir
            
            # Read the last 10 lines of the file to efficiently get the metrics
            try:
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()
                # The metrics are expected to be at the end of the file
                # We take the last 10 lines as a safe buffer
                lines = lines[-10:]
            except FileNotFoundError:
                continue

            # Process each of the last lines to extract metric name and value
            for line in lines:
                line = line.strip()
                if 'Epoch' not in line:
                    metric_name, value_str = line.split(":", 1)
                    try:
                        value = float(value_str.strip())
                        all_metrics.append({
                            'fold': dir,
                            'Metric': metric_name.strip(),
                            'value': value
                        })
                    except ValueError:
                        # Handle cases where the value is not a valid number
                        continue

    # Create the DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # You might want to pivot the table for easier comparison
    if not metrics_df.empty:
        metrics_pivot_df = metrics_df.pivot(index='fold', columns='Metric', values='value')
        return metrics_pivot_df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    # Create output directory if it doesn't exist
    os.makedirs(config['out_path'], exist_ok=True)

    # Check if the data path exists
    if not os.path.exists(config['data_path']):
        print(f"Data path {config['data_path']} does not exist. Please check the path.")
        sys.exit(1)

    # Check if the data file exists
    data_file = os.path.join(config['data_path'], 'hetero_graph.pt')
    if not os.path.isfile(data_file):
        data_file = os.path.join(config['data_path'], 'data.pt')
        if not os.path.isfile(data_file):
            print(f"Data file 'hetero_graph.pt', or 'data.pt', does not exist in {config['data_path']}. Please check the path or file name.")
            sys.exit(1)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("\nRunning configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    # Load preprocessed HeteroData
    print("\nReading data...")
    # data = torch.load(config['data_path'] + '/hetero_graph.pt')
    data = torch.load(data_file)

    about_data(data, config)

    # Adds input nodes, and either train/test masks or split in CV sets
    # (and global negative edges if specified in config['global_neg_edges'])
    process_data(data, config, split = config['split'])

    cv_data = [d for d in os.listdir(config['data_path']) if d.startswith('CV') and os.path.isdir(os.path.join(config['data_path'], d))]
    cv_out = [d for d in os.listdir(config['out_path']) if d.startswith('CV') and os.path.isdir(os.path.join(config['out_path'], d))]

    if len(cv_out) == 0:
        for i in range(len(cv_data)):
            cv_dir = os.path.join(config['out_path'], f'CV{i+1}')
            os.makedirs(cv_dir, exist_ok=True)

        cv_out = [d for d in os.listdir(config['out_path']) if d.startswith('CV') and os.path.isdir(os.path.join(config['out_path'], d))]

    if len(cv_data) != len(cv_out):
        print(f"\nFound {len(cv_out)} existing CV directories at {config['out_path']}. OVERWRITTING with {len(cv_data)} NEW models corresponding to the CV sets at {config['data_path']}.")
        if len(cv_out) > len(cv_data):
            for d in cv_out[len(cv_data):]:
                shutil.rmtree(os.path.join(config['out_path'], d))

        cv_out = cv_out[:len(cv_data)]

    start = time.time()
    processes = []
    print(f"\nStarting training for {len(cv_out)} CV sets...")
    for dir in cv_out:
        data_path = os.path.join(config['data_path'], dir, 'data.pt')
        # model_path = os.path.join(config['out_path'], dir, 'model')
        model_path = os.path.join(config['out_path'], dir)
        log_path = os.path.join(config['out_path'], dir, 'train.log')
        cmd = [
            sys.executable, "run_training.py",
            "--sampling_layers", str(config['sampling_layers']),
            "--num_samples", str(config['num_samples']),
            "--batch_size", str(config['batch_size']),
            "--workers", str(config['workers']),
            "--channels", *(map(str, config['channels'])),
            "--out_channel", str(config['out_channel']),
            "--att_heads", *(map(str, config['att_heads'])),
            "--GNN_layers", str(config['GNN_layers']),
            "--GNN_dropout", str(config['GNN_dropout']),
            "--MLP_dropout", str(config['MLP_dropout']),
            "--epochs", str(config['epochs']),
            "--node_type1", config['node_type1'],
            "--node_type2", config['node_type2'],
            "--rel", config['rel'],
            "--seeded_type", config['seeded_type'],
            # "--ind_MSG_Pass" if config['ind_MSG_Pass'] else "--no-ind_MSG_Pass",
            "--seeded_ind_only" if config['seeded_ind_only'] else "--no-seeded_ind_only",
            '--sup_split', str(config['sup_split']),
            "--balanced" if config['balanced'] else "--not-balanced",
            "--data_path", data_path,
            "--model_path", model_path,
            "--Residuals" if config['Residuals'] else "--no-Residuals",
        ]
        with open(log_path, "w") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            processes.append(p)
        print(f"Launching subprocess {p.pid}: {' '.join(cmd)} > {log_path}")
    print()
    for p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Subprocess {p.pid} failed with return code {p.returncode}. Check the log file for details.")
            print(f"Run time: {(time.time() - start) / 60:.2f} minutes.")
            # sys.exit(1)
        else:
            print(f"Subprocess {p.pid} completed successfully.")

    metrics_dataframe = collect_metrics(config['out_path'])

    if not metrics_dataframe.empty:
        print("\nMetrics:")
        print(metrics_dataframe)
        
        # Example of how you can use the DataFrame
        print("\nMean and Standard Deviation of Metrics across folds:")
        print(metrics_dataframe.agg(['mean', 'std']))
    else:
        print("\nNo metric files found or DataFrame is empty.")

    print(f"\nTotal run time: {(time.time() - start) / 60:.2f} minutes.")