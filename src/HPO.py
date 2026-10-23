import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
# from joblib import Parallel, delayed
from utils_metrics import write_to_out
from run_training import run 
from utils_sampling_processing import process_data
import torch
import os
import argparse
import time
import warnings
import pandas as pd
import re
from pathlib import Path
import json
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='HPO for HGTDR')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--metric', type=str, default='Val_Loss', help='Metric to optimize during HPO, must be one of AUROC, AUPR, or Val Loss')
    parser.add_argument('--n_calls', type=int, default=50, help='Number of hyperparameter configurations to evaluate during HPO')
    parser.add_argument('--parallel', action='store_true', help='Run CVs in parallel')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Do not run CVs in parallel')
    parser.set_defaults(parallel=False)

    parser.add_argument('--sampling_layers', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=314)
    parser.add_argument('--epochs', type=int, default=300)

    parser.add_argument('--node_type1', type=str, default='drug')
    parser.add_argument('--node_type2', type=str, default='disease')
    parser.add_argument('--rel', type=str, default='indication')
    parser.add_argument('--seeded_type', type=str, default='disease', help='Type of node to be seeded')

    return parser.parse_args()

# SPACE = [
#     Categorical([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], name='sup_split'),
#     Categorical([64, 72, 96, 128], name='channels'),
#     Integer(2, 3, name='GNN_layers'),
#     Categorical([4, 8], name='att_heads'),
#     Categorical([64, 72, 96, 128], name='out_channel'),
#     Categorical([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], name='GNN_dropout'),
#     Categorical([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], name='MLP_dropout'),
#     Categorical([True, False], name='seeded_ind_only')
# ]

def load_hpo_space(file_path, fixed_args):
    """
    Loads HPO space from JSON. 
    If a parameter has only one possible value, it is moved directly 
    into fixed_args. Otherwise, it is added to the HPO SPACE list.
    """
    with open(file_path, 'r') as f:
        hpo_data = json.load(f)

    space = []
    for name, params in hpo_data.items():
        stype = params['type'].lower()

        is_fixed = False
        fixed_val = None

        if stype == 'categorical' and len(params['values']) == 1:
            is_fixed, fixed_val = True, params['values'][0]
        elif (stype in ['integer', 'real']) and params['low'] == params['high']:
            is_fixed, fixed_val = True, params['low']

        if is_fixed:
            # Dynamically add/update the attribute on the fixed_args Namespace
            setattr(fixed_args, name, fixed_val)
            continue

        # Otherwise, add to the HPO search space
        if stype == 'categorical':
            space.append(Categorical(params['values'], name=name))
        elif stype == 'integer':
            space.append(Integer(params['low'], params['high'], name=name))
        # elif stype == 'real':
        #     prior = params.get('prior', 'uniform')
        #     space.append(Real(params['low'], params['high'], prior=prior, name=name))

    return space, fixed_args

def initial_points(results_path, SPACE):
    if os.path.exists(results_path):
        print(f"\nReading existing results file at {results_path}:")

        with open(results_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None, None
            # entries = lines[1:-3]

            x0 = []
            y0 = []
            for line in lines[1:-3]:
                vals = line.strip().split()

                # hyperparameters = [p for p in vals[:-1]]
                # hyperparameters = [float(p) if p not in ['True', 'False'] else (p == 'True') for p in vals[:-1]]
                hyperparameters = []
                for dim, p in zip(SPACE, vals[:-3]):
                    if isinstance(dim, Categorical):
                        if dim.name in ['seeded_ind_only']:
                            hyperparameters.append(p == 'True')
                        elif dim.name in ['sup_split', 'GNN_dropout', 'MLP_dropout']:
                            hyperparameters.append(float(p))
                        else:
                            hyperparameters.append(int(p))
                    elif isinstance(dim, Integer):
                        hyperparameters.append(int(p))
                    elif isinstance(dim, Real):
                        hyperparameters.append(float(p))
                # for p in vals[:-1]:
                #     if p in ['True', 'False']:
                #         hyperparameters.append(p == 'True')
                #     elif '.' in p:
                #         hyperparameters.append(float(p))
                #     else:
                #         hyperparameters.append(int(p))
                x0.append(hyperparameters)

                if args.metric == 'Val_Loss':
                    y0.append(float(vals[-3]))
                elif args.metric == 'AUPR':
                    y0.append(float(vals[-2])*-1)
                elif args.metric == 'AUROC':
                    y0.append(float(vals[-1])*-1)

        for params, val in zip(x0, y0):
            print(f"Params: {params} => {args.metric}: {val}")

        with open(results_path, 'w') as f:
            for line in lines[:-3]:  # rewrite without last 3 summary lines
                f.write(line)

    else:
        x0 = y0 = None

        with open(results_path, 'w') as f:
            for dim in SPACE:
                f.write(f"{dim.name}\t")
            f.write(f"Val_Loss\tAUPR\tAUROC\n")

    return x0, y0

# @use_named_args(SPACE)
def cv_wrapper(fixed_args, **hyperparams_dict):
    """
    Wrapper function to pass skopt's list of hyperparameter values
    into the 'config' dict and provide avg of cv scores.
    """
    # 1. Read CV splits
    DATA_DIR = fixed_args.data_path
    cvs = [d for d in os.listdir(DATA_DIR) if d.startswith('CV') and os.path.isdir(os.path.join(DATA_DIR, d))]

    # 2. run CV folds (not in parallel, gpu probs)
    config = hyperparams_dict.copy()
    # config['metric'] = fixed_args.metric

    # config['epochs'] = fixed_args.epochs
    # config['data_path'] = DATA_DIR
    # config['sampling_layers'] = fixed_args.sampling_layers
    # config['num_samples'] = fixed_args.num_samples
    # config['batch_size'] = fixed_args.batch_size

    # config['node_type1'] = fixed_args.node_type1
    # config['node_type2'] = fixed_args.node_type2
    # config['rel'] = fixed_args.rel
    # config['seeded_type'] = fixed_args.seeded_type
    # # config['out_path'] = '../out_testy/hyperparam_opt'
    # # config['seeded_ind_only'] = False
    config['workers'] = 0
    config['Residuals'] = True
    config['balanced'] = False

    for name, value in vars(fixed_args).items(): config[name] = value

    print(f"\nEvaluating hyperparameters: \n{hyperparams_dict}\n")

    if fixed_args.parallel:
        def process_fold(config, dir_name, data_dir):
            data_path = os.path.join(data_dir, dir_name, "data.pt")
            metrics = run(config, data_path, ValAndTest=False)
            print(f"\tFold {dir_name} {config['metric']}: {metrics.loc[metrics.index == config['metric'], 'Value'].values[0]}\n")
            return metrics

        # Run the folds in parallel
        metrics_list = Parallel(n_jobs=len(cvs))(
            delayed(process_fold)(config, dir_name, DATA_DIR) for dir_name in cvs
        )

    else:
        metrics_list = []
        for dir in cvs:
            data_path = os.path.join(DATA_DIR, dir, "data.pt")
            print(f"\tCV Fold at {os.path.join(DATA_DIR, dir)}")
            metrics = run(config, data_path, ValAndTest = False)#, device)
            metrics_list.append(metrics)
            print(f"\tFold {dir} {config['metric']}: {metrics.loc[metrics.index == config['metric'], 'Value'].values[0]}\n")

    metrics = pd.concat(metrics_list, axis=1).mean(axis=1)

    # results_path = fixed_args.out_path + '/HPO_results.txt'
    with open(fixed_args.results_path, 'a') as f:
        for param in hyperparams_dict:
            f.write(f"{hyperparams_dict[param]}\t")
        f.write(f"{metrics.loc[metrics.index == 'Val_Loss', ].values[0]:.4f}\t")
        f.write(f"{metrics.loc[metrics.index == 'AUPR', ].values[0]:.4f}\t")
        f.write(f"{metrics.loc[metrics.index == 'AUROC', ].values[0]:.4f}\n")

    avg_metric = metrics.loc[metrics.index == config['metric'], ].values[0]
    # gp_minimize minimizes the objective, so return the loss or negative metric
    if fixed_args.metric != 'Val_Loss':
        avg_metric = -avg_metric

    return round(avg_metric, 4)

if __name__ == "__main__":

    print("\nStarting hyperparameter optimization with Bayesian Optimization (GP)...")

    args = parse_args()

    DATA_DIR = args.data_path
    config = {
        # "node_type1": args.node_type1,
        # "node_type2": args.node_type2,
        # "rel": args.rel,
        # "seeded_type": args.seeded_type,
        # "data_path": DATA_DIR,

        # "sampling_layers": args.sampling_layers,
        # "num_samples": args.num_samples,
        # "batch_size": args.batch_size,
        # "epochs": args.epochs,
        "workers": 0,
        "Residuals": True,
        "balanced": False
    }
    for name, value in vars(args).items(): config[name] = value

    dir_name = Path(DATA_DIR).name
    data_path = DATA_DIR + '/data.pt'
    inner_cv = re.search(r"CV_?\d+$", dir_name)
    if inner_cv:
        print(f"\nPerforming nested-CV on {DATA_DIR}.\nPartitioning {data_path} into 3-inner CV sets for inner loop HPO")
        data_CV = torch.load(data_path, weights_only=False)
        process_data(data_CV, config, [.333])
    else:
        print(f"\nCurrent dir {DATA_DIR} does not appear to be a CV dir. Performing FINAL HPO")

    print(f"\nStarting Bayesian Optimization")
    start_time = time.time()

    if not os.path.exists(args.out_path): os.makedirs(args.out_path)

    space_file = "SpaceHPO_nested.json" if inner_cv else "SpaceHPO_final.json"
    SPACE, args = load_hpo_space(space_file, args)
    for name, value in vars(args).items(): config[name] = value
    print("\nRunning configuration:")
    for key, value in config.items(): print(f"\t{key}: {value}")

    results_path = args.out_path + '/HPO_results.txt'
    args.results_path = results_path
    x0, y0 = initial_points(results_path, SPACE)

    objective = lambda x: cv_wrapper(
        fixed_args=args, 
        **{dim.name: value for dim, value in zip(SPACE, x)}
    )

    res_gp = gp_minimize(objective,
                        SPACE,
                        n_calls=args.n_calls,
                        x0=x0,
                        y0=y0,
                        n_initial_points=0 if x0 is not None else 10,
                        random_state=42,
                        verbose=True,
                        n_jobs=1)

    with open(results_path, 'a') as f:
        f.write(f"\n{args.metric} Best score: {res_gp.fun}\n")
        for val in res_gp.x:
            f.write(f"{val}\t")

    print(f"\nTotal HPO time: {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"\nBest hyperparameters found for {DATA_DIR}:")

    write_to_out(f"Training final model with best hyperparameters found for {data_path}:", args.out_path)
    for name, val in zip([dim.name for dim in SPACE], res_gp.x):
        print(f"\t{name}: {val}")
        write_to_out(f"{name}: {val}", args.out_path)
        config[name] = val

    data = torch.load(data_path, weights_only=False)
    mask = data[(config['node_type1'], config['rel'], config['node_type2'])].mask
    edges = data[(config['node_type1'], config['rel'], config['node_type2'])].num_edges
    print(f"\nTraining final model with best hyperparameters found for {data_path}...")
    print(f"\tTrain edges: {sum(mask == 0) / edges * 100:.2f}")
    print(f"\tValidation edges: {sum(mask == 1) / edges * 100:.2f}")
    print(f"\tHoldout Test edges: {sum(mask == 2) / edges * 100:.2f}")

    start_time = time.time()
    final_val_metric = run(config, data_path, save_path=args.out_path, final=True, ValAndTest=False)
    print(f"\nFinal model training time: {(time.time() - start_time) / 60:.2f} minutes.")

    # TO DO:
    # deal also with _final.json CHECK IT WORKS
    # make sure the final, ValAndTest logic (i think inside process_batch) is still valid. both for neste-cv and final hpo SEEMS SO

    # this might work too for the FINAL FINAL model after choosing ULTIMATE HPs from FINAL HPO, as long as val -> 0 & test -> 1, so train is done on 90% and test is used for val/test ONCE at the VERY END:
    # read data from any CV
    # edge_mask[edge_mask == 2] = 1
    # edge_mask[edge_mask == 1] = 0
    # save to DATA_DIR
    # run(final=True)