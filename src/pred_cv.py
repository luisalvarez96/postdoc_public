import os
import sys
import subprocess
import argparse
import time
from itertools import combinations
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    parser.add_argument('--Residuals', action='store_true', help='Use residual connections')
    parser.add_argument('--no-Residuals', dest='Residuals', action='store_false', help='Do not use residual connections')
    parser.set_defaults(Residuals=True)

    parser.add_argument('--runs', type=int, default=100, help='Number of runs for prediction')
    parser.add_argument('--graph_samples', action='store_true', help='Use graph sampling for prediction')
    parser.add_argument('--no-graph_samples', dest='graph_samples', action='store_false', help='Do not use graph sampling')
    parser.set_defaults(graph_samples=True)

    parser.add_argument('--node_type1', type=str, default='drug')
    parser.add_argument('--node_type2', type=str, default='disease')
    parser.add_argument('--rel', type=str, default='indication')
    parser.add_argument('--seeded_type', type=str, default='disease')  # Type of node to be seeded

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--out_path', type=str, default='../out')
    return parser.parse_args()

def overlap_matrix(ax, drugs_dict, cv_out, file):

    matrix_size = len(cv_out)
    comparison_matrix = np.full((matrix_size, matrix_size), np.nan)
    for cvA, cvB in combinations(cv_out, 2):
        shared_entries = set(drugs_dict[cvA]['Drug name']).intersection(set(drugs_dict[cvB]['Drug name']))
        den = min(len(drugs_dict[cvA]), len(drugs_dict[cvB]))

        A = re.search(r'\d+', cvA).group()
        B = re.search(r'\d+', cvB).group()

        A = int(A) - 1
        B = int(B) - 1

        if den != 0:
            comparison_matrix[A, B] = len(shared_entries) / den
            comparison_matrix[B, A] = comparison_matrix[A, B]
        else:
            comparison_matrix[A, B] = comparison_matrix[B, A] = 0


    comparison_matrix = np.triu(comparison_matrix, k = 1)
    comparison_matrix[comparison_matrix == 0] = np.nan
    mn = np.nanmean(comparison_matrix)


    im = ax.imshow(comparison_matrix, cmap='YlGnBu', vmin=0.5, vmax=1)
    ax.set_title(f'{file} (avg = {mn:.2f})')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i < j:  # Only add text for the upper triangular part
                ax.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                        ha='center', va='center', color='black', fontsize = 12)
            elif i == j:
                cv = 'CV'+str(i+1)
                ax.text(i, j, f'{len(drugs_dict[cv])}',
                        ha='center', va='center', color='black', fontsize = 12)

    return im

def overlap_figure(base_dir, drugs_perGeneSet_perCV, cv_out):
    num_plots = len(drugs_perGeneSet_perCV)
    fig, axs = plt.subplots(1, num_plots + 1, figsize=(18, 6), gridspec_kw={'width_ratios': [1] * num_plots + [0.05]})

    for i, (title, cv_dict) in enumerate(drugs_perGeneSet_perCV.items()):
        im = overlap_matrix(axs[i], cv_dict, cv_out, title)

    fig.colorbar(im, cax=axs[num_plots], label='Percentage (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'pred_overlap'), dpi=200)

def get_approved_drugs(file_path, cutoff = 100, prt = False):
    GeneSet = Path(file_path).stem.split('_')[-1]
    print(f'\nReading drugs from {GeneSet}...') if prt else None

    df = pd.read_csv(file_path, sep='\t')

    print(f"\tApproved drugs: {sum(df['Approved'])} ({sum(df['Approved'])/len(df)*100:.2f}%)") if prt else None
    df = df[df['Approved']]

    if cutoff:
        print(f"\tAbove Count cutoff ({cutoff}): {sum(df['Count'] >= cutoff)} ({sum(df['Count'] >= cutoff)/len(df)*100:.2f}%)") if prt else None
        df = df[df['Count'] >= cutoff]

    return df, GeneSet

def pred_overlap(base_dir = "../out", cutoff = 0, save = True):
    cv_out = [d for d in os.listdir(base_dir) if d.startswith('CV') and os.path.isdir(os.path.join(base_dir, d))]


    drugs_perGeneSet_perCV = {}

    for cv in cv_out:
        print(f'\nReading predictions in {cv}:')
        cv_dir_path = os.path.join(base_dir, cv)

        data_sets = []
        titles = []
        for file in os.listdir(cv_dir_path):
            if file.endswith('.csv'):
                df, GeneSet = get_approved_drugs(os.path.join(cv_dir_path, file), cutoff=cutoff, prt=True)
                data_sets.append(df)
                titles.append(GeneSet)


        labeled = 'Class' in df
        if labeled:
            print(f'Predictions {GeneSet} for {cv}:')

            print('AT:')
            print('\tInd:', sum(df['Class'] == 1))
            print('\tContra:', sum(df['Class'] == 2))
            print('\t-1:', sum(df['Class'] == -1))


        if not labeled:

            for (data, title) in zip(data_sets, titles):
                drugs_perGeneSet_perCV.setdefault(title, {})[cv] = data[['Drug name', 'Count', 'Prob']]

        else:

            for (data, title) in zip(data_sets, titles):
                drugs_perGeneSet_perCV.setdefault(title, {})[cv] = data[['Drug name', 'Count', 'Prob']][data['Class'] == 1]

    # Finding shared drugs across all CVs

    shared_drugs = {}
    for title, cv_dict in drugs_perGeneSet_perCV.items():
        drug_names = set.intersection(*(set(df['Drug name']) for df in cv_dict.values()))
        shared_drugs[title] = pd.DataFrame(list(drug_names), columns=['Drug name'])

        for cv, df_cv in cv_dict.items():
            shared_drugs[title] = shared_drugs[title].merge(df_cv[['Count', 'Prob', 'Drug name']], on='Drug name', how='left').rename(columns={'Count': 'Count'+cv, 'Prob': 'Prob'+cv})

        cv_cols = shared_drugs[title].filter(regex=r'^ProbCV\d+')
        shared_drugs[title]['ProbAvg'] = cv_cols.mean(axis=1).round(2)
        shared_drugs[title]['ProbStd'] = cv_cols.std(axis=1).round(2)
        shared_drugs[title]['CountAvg'] = shared_drugs[title].filter(regex=r'^CountCV\d+').mean(axis=1).round(2)

        shared_drugs[title].sort_values(by=['ProbAvg', 'CountAvg'], ascending=[False, False], inplace=True)

        if save:
            shared_drugs[title][['Drug name', 'ProbAvg', 'ProbStd', 'CountAvg']].to_csv(os.path.join(base_dir, f'shared_drugs_{title}.csv'), sep='\t', index=False, header=True)
            print(f'\n{title} shared predictions saved to {os.path.join(base_dir, f"shared_drugs_{title}.csv")}')

        print(f'\n{title} shared predictions: {len(shared_drugs[title])}')


    if save:

        overlap_figure(base_dir, drugs_perGeneSet_perCV, cv_out)


def plt_hists(base_dir, cutoff = 100):
    cv_out = [d for d in os.listdir(base_dir) if d.startswith('CV') and os.path.isdir(os.path.join(base_dir, d))]

    fig, axes = plt.subplots(len(cv_out), 3, figsize=(15, 18), sharey=True, sharex=True)

    for row, cv in enumerate(cv_out): 
        AT = pd.read_csv(base_dir + f'/{cv}' + '/nipah_drugs_AT.csv', sep='\t')
        endos = pd.read_csv(base_dir + f'/{cv}' + '/nipah_drugs_endos.csv', sep='\t')
        MIRs = pd.read_csv(base_dir + f'/{cv}' + '/nipah_drugs_MIRs.csv', sep='\t')

        AT = AT[AT['Approved']]
        endos = endos[endos['Approved']]
        MIRs = MIRs[MIRs['Approved']]

        AT = AT[AT['Count'] > cutoff]
        endos = endos[endos['Count'] > cutoff]
        MIRs = MIRs[MIRs['Count'] > cutoff]

# --- Plotting on the Axes Grid ---
        data_sets = [AT, endos, MIRs]
        colors = ['skyblue', 'aqua', 'lightgreen']
        titles = ['AT', 'endos', 'MIRs']
        
        for col, (data, color, title) in enumerate(zip(data_sets, colors, titles)):
            ax = axes[row, col]

            ax.hist(data['Prob'], bins=100, color=color)#, edgecolor='black', alpha=1)
            ax.grid(axis='y', alpha=1)
            ax.grid(axis='x', alpha=1)

            # Set title for each subplot
            if row == 0:
                ax.set_title(f'{title}', fontsize=24)

            # Label y-axis only for the first column
            if col == 0:
                ax.set_ylabel(f'{cv}', fontsize=20)
            
            # Label x-axis only for the bottom row
            if row == len(cv_out) - 1:
                ax.set_xlabel('Probability', fontsize=15)

    plt.savefig(base_dir + '/PredProbHist.png')

if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    print("\nRuning predictions with Configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    # Check if the data path exists
    if not os.path.exists(config['data_dir']):
        print(f"Data path {config['data_dir']} does not exist. Please check the path.")
        sys.exit(1)

    # Check if the CV sets path exists
    if not os.path.exists(config['out_path']):
        print(f"Out path {config['out_path']} does not exist. Please check the path.")
        sys.exit(1)

    cv_dirs = [d for d in os.listdir(config['out_path']) if d.startswith('CV') and os.path.isdir(os.path.join(config['out_path'], d))]

    if not cv_dirs:
        print(f"\nNo CV directories found in {config['out_path']}.")

        if os.path.isfile(os.path.join(config['out_path'], 'model.h5')):
            print(f'\nFound model.h5 in {config["out_path"]}. Running predictions with it...')
        else:
            print(f'No model.h5 found in {config["out_path"]}. Please check the path or file name.')
            sys.exit(1)

    start = time.time()
    processes = []
    print(f"\nFound {len(cv_dirs)} CV directories in {config['out_path']}. Running predictions...")
    for cv_dir in cv_dirs:
        cv_path = os.path.join(config['out_path'], cv_dir)
        if not os.path.isdir(cv_path):
            print(f"\n{cv_path} is not a directory. Skipping...")
            continue

        # Check if the model file exists in the CV directory
        model_file = os.path.join(cv_path, 'model.h5')
        if not os.path.isfile(model_file):
            print(f"\nModel file {model_file} does not exist in {cv_path}. Skipping...")
            continue

        # Run the prediction script for each CV directory

        log_path = os.path.join(cv_path, 'pred.log')
        cmd = [
            sys.executable, 'nipah_pred.py',
            '--sampling_layers', str(config['sampling_layers']),
            '--num_samples', str(config['num_samples']),
            '--batch_size', str(config['batch_size']),
            '--workers', str(config['workers']),
            "--channels", *(map(str, config['channels'])),
            "--out_channel", str(config['out_channel']),
            "--att_heads", *(map(str, config['att_heads'])),
            "--GNN_layers", str(config['GNN_layers']),
            "--GNN_dropout", str(config['GNN_dropout']),
            "--MLP_dropout", str(config['MLP_dropout']),
            '--node_type1', config['node_type1'],
            '--node_type2', config['node_type2'],
            '--rel', config['rel'],
            "--seeded_type", config['seeded_type'],
            '--data_dir', config['data_dir'],
            '--model_dir', cv_path,
            '--runs', str(config['runs']),
            '--graph_samples' if config['graph_samples'] else '--no-graph_samples',
            '--Residuals' if config['Residuals'] else '--no-Residuals'
        ]

        with open(log_path, 'w') as log_file:
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append(p)
        print(f"Launching subprocess {p.pid}: {' '.join(cmd)} > {log_path}")

    print()

    for p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Subprocess {p.pid} failed with return code {p.returncode}. Check the log file for details.")
            print(f"Run time: {(time.time() - start) / 60:.2f} minutes.")
        else:
            print(f"Subprocess {p.pid} completed successfully.")

    pred_overlap(config['out_path'], cutoff=config['runs'])
    plt_hists(config['out_path'], cutoff=config['runs'])

    print(f"\nTotal run time: {(time.time() - start) / 60:.2f} minutes.")