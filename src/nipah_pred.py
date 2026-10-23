import torch
import pandas as pd
import pickle
import argparse
import sys
import time
import os
import re
from model import *
from utils_sampling_processing import *
from run_training import define_model
from utils_metrics import write_to_out
from run_cv import about_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_layers', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=164)
    parser.add_argument('--channels', nargs='+', type=int, default=[64])
    parser.add_argument('--out_channel', type=int, default=64)
    parser.add_argument('--att_heads', nargs='+', type=int, default=8)
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
    parser.add_argument('--model_dir', type=str, default='../out')
    return parser.parse_args()

def sample_pred(batch, GNN, pred, config, c, nipah_id = 6780):
    rel = config['rel']
    node_type1 = config['node_type1']
    node_type2 = config['node_type2']

    labeled = 'edge_label' in batch[(node_type1, rel, node_type2)]
    if labeled:
        # batch, edges_batch, labels_batch = process_batch(batch, config, False)
        edge_label = batch[(node_type1, rel, node_type2)].edge_label
        if 0 in edge_label:
            batch[(node_type1, rel, node_type2)].edge_index = batch[(node_type1, rel, node_type2)].edge_index[:, edge_label != 0]
            edge_label = edge_label[edge_label != 0]

    if nipah_id in batch[node_type2].id: 

        batch = batch.to(device)
        drug_embeddings, disease_embeddings = GNN(batch.x_dict, batch.edge_index_dict)
        # print('drug_embeddings shape:', drug_embeddings.shape)
        # print('disease_embeddings shape:', disease_embeddings.shape)

        Nipah_idx = torch.where(batch[node_type2].id == nipah_id)[0]

        batch_out = pred(drug_embeddings, disease_embeddings[[Nipah_idx] * drug_embeddings.shape[0]])#[:, 0]

        if len(Nipah_idx) == 0:
            write_to_out("No Nipah node found!", config['model_dir'])
        # else:
        #     print("Nipah_idx:", Nipah_idx)
        #     print("batch_out > 0:", (batch_out > 0).sum())
        #     print("batch_out:", batch_out)

        if batch_out.shape[1] == 1:
            batch_out = batch_out[:, 0]  # For binary classification, take the first column
            if (batch_out > 0).sum() > 0:
                sample_pred_drugs = batch[node_type1].id[batch_out > 0]
                sample_pred_probs = torch.sigmoid(batch_out[batch_out > 0]).cpu().numpy()
                preds = [sample_pred_drugs, sample_pred_probs]
            else:
                preds = [torch.tensor([]), torch.tensor([])]
        else:
            # For multi-class classification, take the class with the highest score
            if (batch_out.argmax(dim=1) != 0).sum() > 0:
                # print("batch_out.argmax:\n", batch_out.argmax(dim=1))

                sample_pred_drugs = batch[node_type1].id[batch_out.argmax(dim=1) != 0]
                sample_pred_class = batch_out.argmax(dim=1)[batch_out.argmax(dim=1) != 0]
                sample_pred_probs = torch.softmax(batch_out, dim=1).max(dim=1).values[batch_out.argmax(dim=1) != 0].cpu().numpy()
                preds = [sample_pred_drugs, sample_pred_probs, sample_pred_class]
            else:
                # print("No indications predicted for Nipah virus.")
                preds = [torch.tensor([]), torch.tensor([]), torch.tensor([])]

    else:
        write_to_out(f'NIPAH NOT IN BATCH {c}!!!', config['model_dir'])
        preds = [torch.tensor([]), torch.tensor([])]
        if labeled: preds = preds + [torch.tensor([])]

    if len(preds[0]) == 1:
        write_to_out(f'{sample_pred_drugs[0]} pred with prob = {sample_pred_probs[0]:.2f}', config['model_dir'])
    elif len(preds[0]) > 1:
        if len(preds) == 2:
            write_to_out(f'More than one indication predicted: {len(sample_pred_drugs)} drugs and {len(sample_pred_probs)} probabilities', config['model_dir'])
        elif len(preds) == 3:
            write_to_out(f'{len(sample_pred_drugs)} drugs with {(sample_pred_class == 1).sum().item()} indications and {(sample_pred_class == 2).sum().item()} contraindications predicted', config['model_dir'])

    if len(preds[0]) != len(preds[1]):
        write_to_out(f'Warning: Number of predicted drugs ({len(sample_pred_drugs)}) does not match number of probabilities ({len(sample_pred_probs)})', config['model_dir'])
#                            sys.exit()
    return preds

def process_results(pred_ind, entity_dictionary, KG, approved):
    inv_drug_dict = {v: k for k, v in entity_dictionary['drug'].items()}

    results = []

    if len(pred_ind) == 0:
        if pred_ind.shape[1] == 3:
            return pd.DataFrame({'G_id': [], 'Index': [], 'Count': [], 'Prob': [], 'Class': [], 'Approved': [], 'Drug name': []})
        else:
            return pd.DataFrame({'G_id': [], 'Index': [], 'Count': [], 'Prob': [], 'Approved': [], 'Drug name': []})

    for idx in pred_ind['drug index'].unique():
        KG_index = inv_drug_dict[idx].split('::')[1]
        drug_names = KG['x_name'][KG['x_index'] == int(KG_index)].unique()
        mean_prob = pred_ind['prob'][pred_ind['drug index'] == idx].mean()
        count = (pred_ind['drug index'] == idx).sum()
        status = drug_names[0] in approved['name'].values

        if pred_ind.shape[1] > 2:  
            drug_classes = pred_ind['class'][pred_ind['drug index'] == idx].unique()
            if len(drug_classes) > 1:
                print(f'Warning: Multiple classes found for drug {drug_names[0]} ({inv_drug_dict[idx]}): {drug_classes}')

            results.append({
                'G_id': idx,
                'Index': KG_index,
                'Count': count,
                'Prob': round(mean_prob * 100, 2),
                'Class': drug_classes[0] if len(drug_classes) == 1 else -1,  # -1 if multiple classes,
                'Approved': status,
                'Drug name': drug_names[0]
            })

        else:

            results.append({
                'G_id': idx,
                'Index': KG_index,
                'Count': count,
                'Prob': round(mean_prob * 100, 2),
                'Approved': status,
                'Drug name': drug_names[0],
            })

    return pd.DataFrame(results)

def pred_nipah_drugs(config, data_path, results_path, KG, entity_dictionary, approved):
    # node_type1 = config['node_type1']

    print(f'\nLoading data from {data_path}...')
    data = torch.load(data_path)

    about_data(data, config)

    nipah_index = KG[KG['x_name'].str.contains('Nipah', regex=True, na=False)]['x_index'].unique()
    nipah_id = entity_dictionary['disease']['disease::' + ','.join(nipah_index.astype(str))]

    # Load the model
    print(f'\nLoading model from {config["model_path"]} at device {device}...')
    GNN, pred, model = define_model(config, data, device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))

    #model.load_state_dict(torch.load('../out/nipah.h5', map_location=device))
    model.eval()  # Set the model to evaluation mode

    if config['graph_samples']:
        # Use a mini-batch from the loader to avoid OOM
        loader = HGTLoader(data, 
                        num_samples=[config['num_samples']] * config['sampling_layers'], 
                        batch_size=config['batch_size'], 
        #                batch_size=164, 
        #                batch_size=data[node_type1].num_nodes, 
                        input_nodes=(config['seeded_type'], None), 
        #                input_nodes=(node_type2, torch.tensor([2675])), 
                        shuffle=True, 
                        num_workers = config['workers'], 
                        persistent_workers = bool(config['workers'])
                        )
        print(f"\nIterating through the loader, for progress check file {config['model_dir']}/out.txt...")

    GNN.eval()
    pred.eval()
    with torch.no_grad():

        pred_drugs = []
        ind_prob = []
        pred_classes = []
        write_to_out(f"\nRunning predictions ({config['runs']} runs) for data at {data_path}", config['model_dir'])
        for run in range(config['runs']):
            write_to_out(f'\nRun {run}', config['model_dir'])
            c = 0

            # Clear CUDA cache to help avoid OOM
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if config['graph_samples']:
                # Iterate through the loader

                for batch in iter(loader):
                    c += 1
    #                print(f'Batch {c}')

                    # selected_drugs, selected_probs = sample_pred(batch, GNN, pred, config)
                    preds = sample_pred(batch, GNN, pred, config, c, nipah_id)

                    pred_drugs.extend(preds[0].tolist())
                    ind_prob.extend(preds[1].tolist())
                    if len(preds) > 2:
                        pred_classes.extend(preds[2].tolist())


            else:
                # Use the entire graph for prediction
                print('Using the entire graph for prediction...')
                selected_drugs, selected_probs = sample_pred(data, GNN, pred, config)
                print(f"pred ran with data with {(data.num_nodes)} nodes and {(data.num_edges)} edges.")


                pred_drugs.extend(selected_drugs.tolist())
                ind_prob.extend(selected_probs.tolist())

                if len(selected_probs) == 1:
                    print(f'{selected_drugs[0]} pred with prob = {selected_probs[0]:.2f}')
                else:
                    print(f'Indications predicted: {len(selected_drugs)} drugs and {len(selected_probs)} probabilities')

    del loader  # Clear the loader to free memory

    print(f'\n{len(pred_drugs)} total predictions in {(run+1)*c} total batches ({c} batches per run).')

    if len(pred_drugs) != len(ind_prob):
        print(f'Warning: Number of predicted drugs ({len(pred_drugs)}) does not match number of probabilities ({len(ind_prob)})')
        return
#        sys.ebut xit()
#        stop()

    # if pred_drugs:
    #     pred_ind = pd.DataFrame({'drug index': pred_drugs, 'prob': ind_prob})
    #     if len(pred_classes) > 0:
    #         pred_ind['class'] = pred_classes

    # else:
    #     print('\nNo drugs predicted for Nipah virus.')
    #     with open(results_path, 'w') as f:
    #         f.write('No drugs predicted for Nipah virus with model at ' + config['model_path'] + ' and data at ' + data_path + '\n')
        
    #     return

    pred_ind = pd.DataFrame({'drug index': pred_drugs, 'prob': ind_prob})
    if len(preds) == 3:
        pred_ind['class'] = pred_classes

    print('\n', pred_ind)

    print('Obtained predictions, now processing results...')

    results = process_results(pred_ind, entity_dictionary, KG, approved)
    results.sort_values(by=['Count', 'Prob'], ascending=[False, False], inplace=True)

    print(f'\nResults:\n{results}')

    # Save the results to a CSV file
    # results_path = config['results_path']
    results.to_csv(results_path, index=False, sep='\t')
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load preprocessed HeteroData
#    data = torch.load('../data/hetero_graph.pt')
    # data = torch.load(config['data_path'])
    # data = process_data(data, config)

    # Load the entity dictionary and KG to map drug indices to names
    # with open('../data/entity_dictionary.pkl', 'rb') as f:
    print("\nLoading entity dictionary...")
    with open(config['data_dir'] + '/entity_dictionary.pkl', 'rb') as f:
        entity_dictionary = pickle.load(f)
    
    print('\nLoading knowledge graph...')
    # KG = pd.read_csv('../data/kg.csv', sep =",")
    KG = pd.read_csv(config['data_dir'] + '/kg.csv', sep =",")

    print('\nLoading approved drugs list...')
    approved = pd.read_csv(config['data_dir'] + '/approved_drugs_inKG.csv', sep ="\t")

    files = os.listdir(config['data_dir'])
    pred_files = [f for f in files if f.endswith('.pt') and 'hetero_graph' in f and f != 'hetero_graph.pt']

    for f in pred_files:
        data_file = os.path.join(config['data_dir'], f)
        suffix = re.search(r'graph(.*)\.pt', f).group(1)
        results_path = config['model_dir'] + '/nipah_drugs' + suffix + '.csv'
        config['model_path'] = config['model_dir'] + '/model.h5'

        print(f'\n\tRunning predictions for {f}...')
        start_time = time.time()
        pred_nipah_drugs(config, data_file, results_path, KG, entity_dictionary, approved)
        elapsed_time = time.time() - start_time
        print(f"\nTotal prediction time for {f}: {elapsed_time / 60:.2f} minutes.\n")



