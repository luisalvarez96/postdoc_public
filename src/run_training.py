from typing import Final
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import HGTLoader
from torch_geometric.data import HeteroData
import torch.nn.functional as F
#import pickle5 as pickle
import pickle
import torch.nn as nn
import pandas as pd
import random
import torch
import copy
import sys
import time
import argparse
from model import *
from utils_sampling_processing import *
from utils_metrics import *
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_layers', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=164) # 360
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

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model_path', type=str, default='../out') # /saved_model')
    parser.add_argument('--data_path', type=str, default='../data') # /hetero_graph.pt')
    
    return parser.parse_args()

def define_model(config, data, device):
    node_type1 = config['node_type1']
    node_type2 = config['node_type2']
    rel = config['rel']    

    labeled = 'edge_label' in data[(node_type1, rel, node_type2)]
    if labeled:
        class_num = data[(node_type1, rel, node_type2)].edge_label.unique()[-1]
        if class_num > 1:
            class_num = class_num + 1  # Adjust for binary classification
            print(f"\nMULTI-CLASS  with class_num = {class_num}")
    else:
        class_num = 1

    # if config['Residuals']:
    #     print("\nUsing RESIDUAL CONNECTIONS in the model.")
    # else:
    #     print("\nNOT using residual connections in the model.")

    if isinstance(config['channels'], (int, float, np.number)):
        config['channels'] = [config['channels']]

    if len(config['channels']) == 1:
        hidden_channels = config['channels'] * int((config['GNN_layers'] + 1))
    elif config['sampling_layers'] != 0 and len(config['channels']) != (config['sampling_layers'] + 1):
        print(f"Hidden channels size {len(config['channels'])} is not appropriate for GNN number of layers {config['GNN_layers']}")
        sys.exit(1)
    else:
        hidden_channels = config['channels']

    if config['sampling_layers'] != 0 and config['GNN_layers'] > config['sampling_layers']:
        print(f"GNN layers ({config['GNN_layers']}) should be <= sampling layers ({config['sampling_layers']})")
        sys.exit(1)

    if isinstance(config['att_heads'], (int, float, np.number)):
        config['att_heads'] = [config['att_heads']]

    if len(config['att_heads']) == 1:
        config['att_heads'] = config['att_heads'] * int(config['GNN_layers'])
    elif len(config['att_heads']) < config['GNN_layers']:
        print(f"Number of attention heads provided should be {config['GNN_layers']}, one for each GNN layer.")
        sys.exit(1)
    elif len(config['att_heads']) > config['GNN_layers']:
        print(f"Only {config['GNN_layers']} entries of att_heads will be used, one for each layer.")

    GNN = HGT(hidden_channels=hidden_channels, #[64, 64, 64, 64],
            #   out_channels=config['out_channel'], #64,
            #   num_heads=config['att_heads'], #[8, 8, 8],
            #   num_layers=config['GNN_layers'], #3,
#              dropout=dropout,
              node_types=data.node_types,
              metadata=data.metadata(),
              config=config)

    # pred = MLPPredictor(64, dropout = 0.2, class_num = class_num)
    pred = MLPPredictor(config['out_channel'], dropout = config['MLP_dropout'], class_num = class_num)
    model = nn.Sequential(GNN, pred)
    model.to(device)
    
    return GNN, pred, model

def shuffle_train_val_splits(data):
    processed_edge_types = set()
    for edge_type in data.edge_types:
        if edge_type[1] in processed_edge_types:
            continue
        else:
            processed_edge_types.add(edge_type[1])

            mask = data[edge_type].mask

            # 1. Identify indices for train (0) and val (1)
            train_val_indices = torch.where((mask != 2))[0]
            # print(f"test edges before: {torch.where((mask == 2))[0][:10]}")

            # 2. Record the original number of training edges to maintain the ratio
            num_train = (mask == 0).sum().item()

            # 3. Shuffle the combined indices
            shuffled_indices = train_val_indices[torch.randperm(train_val_indices.size(0))]

            # 4. Re-assign masks: 
            # First 'num_train' edges become 0, the rest become 1
            new_train_idx = shuffled_indices[:num_train]
            new_val_idx = shuffled_indices[num_train:]

            mask[new_train_idx] = 0
            mask[new_val_idx] = 1
            # print(f"test edges after: {torch.where((mask == 2))[0][:10]}")

            # Update the data object
            data[edge_type].mask = mask

    return data

def epoch(batch, config, GNN, pred, device, stage = 'train'):

    batch, edges_batch, labels_batch = process_batch(batch, config, stage)
    batch = batch.to(device)
    drug_embeddings, disease_embeddings = GNN(batch.x_dict, batch.edge_index_dict)
    c = drug_embeddings[edges_batch[0]]
    d = disease_embeddings[edges_batch[1]]
    out = pred(c, d)#[:, 0]

    return out, labels_batch.to(device)

def restore_edge_index(data):
    processed_edge_types = set()
    for edge_type in data.edge_types:
        if edge_type[1] in processed_edge_types:
            continue
        else:
            processed_edge_types.add(edge_type[1])
            data[edge_type].edge_index = data[edge_type].all_edges
            data[(edge_type[2], edge_type[1], edge_type[0])].edge_index = data[edge_type].edge_index.flip(0)

    return data

def train(GNN, pred, model, loader, optimizer, config, device):
    model.train()
    # if isinstance(loader, Iterable):
    if config['sampling_layers'] > 0:

        total_examples = total_loss = 0
        for i, batch in enumerate(iter(loader)):
            optimizer.zero_grad()

            out, labels_batch = epoch(batch, config, GNN, pred, device)
            loss = compute_loss(out, labels_batch)
            loss.backward()
            optimizer.step()

            total_examples += labels_batch.shape[0]
            total_loss += float(loss) * labels_batch.shape[0]

        return total_loss / total_examples

    else:

        loader = restore_edge_index(loader)

        out, labels_batch = epoch(loader, config, GNN, pred, device)
        loss = compute_loss(out, labels_batch)
        loss.backward()
        optimizer.step()

        return float(loss)

@torch.no_grad()
def val_test(GNN, pred, model, loader, stage, config, device):
    model.eval()

    # if isinstance(loader, Iterable):
    if config['sampling_layers'] > 0:
        total_examples = total_correct = 0
        out, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
        # source, dest = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for batch in iter(loader):

            batch_out, labels_batch = epoch(batch, config, GNN, pred, device, stage)
            labels = torch.cat((labels, labels_batch.to(device)))
            out = torch.cat((out, batch_out))

    else:
        loader = restore_edge_index(loader)

        out, labels = epoch(loader, config, GNN, pred, device, stage)

    loss = compute_loss(out, labels)
    # return out, labels, source, dest, loss.cpu().numpy()
    return out, labels, loss.cpu().numpy()


def run(config, data_path, save_path = None, final = False, ValAndTest = True): #ValAndTest = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('\ndevice:', device)

    data = torch.load(data_path, weights_only=False)
    if 'mask' not in data[(config['node_type1'], config['rel'], config['node_type2'])]:
        print('\nProcessing loaded data to add masks...')
        data = process_data(data, config, split = [.2, .1], cross_validate = False)  # No CV sets, just add masks
        print('Data processed...')
    # else:
    #     print('\nLoaded data already processed with masks, proceeding with training...')

    # This was to try global supervision mask
    # num_inds = data[('drug', 'indication', 'disease')].num_edges
    # supervision = torch.zeros(num_inds, dtype=torch.bool)
    # supervision[random.sample(range(num_inds), int(num_inds*0.8))] = True
    # data[('drug', 'indication', 'disease')].supervision = supervision
    # data[('disease', 'indication', 'drug')].supervision = supervision

#    train_loader, val_loader = define_loaders(config)
    if config['sampling_layers'] > 0:
        loader = sampling(config, data, config['workers'])
    else:
        processed_edge_types = set()
        for edge_type in data.edge_types:
            if edge_type[1] in processed_edge_types:
                continue
            else:
                processed_edge_types.add(edge_type[1])
                data[edge_type].all_edges = data[edge_type].edge_index
                data[(edge_type[2], edge_type[1], edge_type[0])].all_edges = data[edge_type].edge_index.flip(0)

        loader = data
        # config['seeded_ind_only'] = True


    GNN, pred, model = define_model(config, data, device)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=config['epochs'], 
                                                           eta_min=0, 
                                                           last_epoch=-1)
    # warnings.filterwarnings("ignore", category=UserWarning, 
    #         message="The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.",
    #         # You can be even more specific by checking the module it came from
    #         module='torch.optim.lr_scheduler'
    # )

    # print(f'\nStarting epochs iteration, for progess see the output file {os.path.dirname(save_path)}/out.txt')
    # print('\nStarting epochs iteration...')

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path is not None:
        print(f'for progess see the output file {save_path}/out.txt\n')
        if not final and os.path.exists(save_path + '/out.txt'):
            os.remove(save_path + '/out.txt')

    losses, val_losses = [], []
    for epoch in range(config['epochs']):
        if config['sampling_layers'] == 0:
            loader = shuffle_train_val_splits(loader)
        loss = train(GNN, pred, model, loader, optimizer, config, device)
        losses.append(loss)
        # print(f'train loss, epoch {epoch}: {loss}')
        if not final or ValAndTest:
            # out, labels, source, dest, val_loss = val_test(GNN, pred, model, loader, stage = 'val')
            out, labels, val_loss = val_test(GNN, pred, model, loader, 'val', config, device)
            val_losses.append(val_loss)
            txt_to_out = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ValLoss: {val_loss:.4f}'
        else:
            txt_to_out = f'Epoch: {epoch:02d}, Loss: {loss:.4f}'
        # print(txt_to_out)
        if save_path is not None:
            write_to_out(txt_to_out, save_path)
            plot_losses(losses, val_losses, save_path)

        scheduler.step()

    if save_path is not None:
        torch.save(model.state_dict(), save_path + '/model.h5')
        print(f"Model saved to {save_path}/model.h5\n")

    if final or ValAndTest:
        stage = 'test' if ValAndTest else 'val'
        loss = 'Test_Loss' if ValAndTest else 'Val_Loss'
        out, labels, test_loss = val_test(GNN, pred, model, loader, stage, config, device)
        model_metrics = [{'Metric': loss, 'Value': test_loss}]
        if save_path is not None:
            write_to_out(f'{loss}: {test_loss:.4f}', save_path)

    else:
        model_metrics = [{'Metric': 'Val_Loss', 'Value': val_loss}]

    aupr_metrics = AUPR(out, labels, save_path)
    auroc_metrics = AUROC(out, labels, save_path)
    
    model_metrics = model_metrics + aupr_metrics + auroc_metrics
    model_metrics = pd.DataFrame(model_metrics).set_index('Metric')

    if final or ValAndTest:
        print("Metrics:")
        # print(f'Test Loss: {test_loss:.4f}')
        print(model_metrics)

# --- CLEANUP SECTION ---

    # DataLoader Cleanup
    # if loader.num_workers > 0:
    #     loader._shutdown_workers() 
    del loader

    # Model Cleanup
    del GNN
    del pred
    del model

    # VRAM Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_metrics


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("\nRunning configuration:")
    for key, value in config.items():
        print(f"\t{key}: {value}")

    print(f"\nStarting training for dataset at {config['data_path']}...\n")

    # print(f"\nTraining CV split in {os.path.join(config['out_path'], dir)}...")
    start_time = time.time()
    run(config, config['data_path'], save_path = config['model_path'])
    # print(f"Finished training with CV split in {os.path.join(config['out_path'], dir)}.")
    print(f"\nTraining completed for dataset at {config['data_path']} in {(time.time() - start_time) / 60:.2f} minutes.")

    # Run the training process  
    # start_time = time.time()
    # run(config, config['data_path'], save_path = config['model_path'])
    # print(f"\nTraining completed in {(time.time() - start_time) / 60:.2f} minutes.\n")