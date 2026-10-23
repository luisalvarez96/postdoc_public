from networkx import edges
import torch
from torch_geometric.loader import HGTLoader
import random
import copy
import os
import shutil
import sys

# Generate negative edges
def make_neg_edges(edge_index, length):
    drugs = edge_index[0].unique()
    diseases = edge_index[1].unique()
    existing_edges = set(map(tuple, edge_index.t().tolist()))

    neg_edges_source, neg_edges_dest = [], []
    while len(neg_edges_source) < length:
        source = drugs[torch.randint(0, len(drugs), (1,))].item()
        dest = diseases[torch.randint(0, len(diseases), (1,))].item()
        neg_edge = (source, dest)
        if neg_edge not in existing_edges:
#            if len(neg_edges_source) % 1000 == 0:
#                print(f"Adding {len(neg_edges_source)} negative edge")
            neg_edges_source.append(source)
            neg_edges_dest.append(dest)
            existing_edges.add(neg_edge)
    neg_edges = torch.tensor([neg_edges_source, neg_edges_dest])

    return neg_edges

def sampling(config, data, workers = 0):
    loader = HGTLoader(data, 
                   num_samples=[config['num_samples']] * config['sampling_layers'], 
                   batch_size=config['batch_size'], 
                #    input_nodes=(config['node_type1'], data[config['node_type1']]['input_nodes']), 
                   input_nodes=(config['seeded_type'], data[config['seeded_type']].input_nodes), 
                   shuffle=True, 
                   num_workers = workers, 
                   persistent_workers = bool(workers)
                   )
    return loader

def data_split(data, config, split = [.2], cross_validate = True):
    if not cross_validate:
        print('NO CROSS-VALIDATION. Adding train/val/test mask and RETURNING DATA...')
        if len(split) != 2:
            raise ValueError(f"Incorrect side for split, val and test percentage needed to run without cross-validation.")

        print(f"Splitting data into {int((1 - sum(split)) * 100)}-{int(split[0] * 100)}-{int(split[1] * 100)} train-validation-test split")
        processed_edge_types = set()
        for edge_type in data.edge_types:
            if edge_type[1] in processed_edge_types:
                continue
            else:
                processed_edge_types.add(edge_type[1])
                # Assign mask for all edges, MSG-PASS & supervision (indications)
                edge_mask = torch.zeros(data[edge_type].num_edges)

                indices = list(range(data[edge_type].num_edges))
                # random.shuffle(indices)
                # edge_mask = edge_mask[indices]

                val_end = int(len(indices) * split[0])
                edge_mask[:val_end] = 1
                edge_mask[val_end:(val_end + int(len(indices) * split[1]))] = 2

                random.shuffle(indices)
                edge_mask = edge_mask[indices]
                data[edge_type]['mask'] = edge_mask
                data[edge_type[2], edge_type[1], edge_type[0]]['mask'] = edge_mask

        return data

    else:
        # --- CV sets logic ---

        if len(split) > 2:
            raise ValueError(f"Invalid length for split: {len(split)}. Must be 1 (for validation) split or 2 (for validation-test) splits.")
        if split[0] > 1:
            raise ValueError(f"Split percentage must be given as a fraction < 1.")

        print('\nMaking CV split\n')
        masked = 'mask' in data[(config['node_type1'], config['rel'], config['node_type2'])]
        if masked:
            edge_mask = data[(config['node_type1'], config['rel'], config['node_type2'])]['mask']
            print(f"Data already masked. Mask values: {edge_mask.unique()}")

            if len(split) == 2 and 1 in edge_mask:
                print("Data already proceesed and split of length 2 provided! Unclear instructions.")
                sys.exit(1)
            else:
                if len(edge_mask.unique()) == 2:
                    print("Proceeding with nested cv!")
                elif len(edge_mask.unique()) == 3:
                    print("Data already proceesed with val and test masks!! Adding val mask to test mask for nested CV split!")
                    # sys.exit(1)

        if len(split) == 2:
            split_ptg = (1 - split[1])

        else:
            split_ptg = 1

        CV_sets = int(split_ptg / split[0])

        # print(f"Splitting {total_edges - test_edges} train edges into {CV_sets}-fold CV sets, {int(split[0] * 100)}% for validation per fold.")
        print(f"Splitting into {CV_sets}-fold CV sets.")
        if len(split) == 2:
            print(f"\tSplitting into {int((1-split[1]-split[0]) * 100)}-{int(split[0] * 100)}-{int(split[1] * 100)} train-validation-test split")
            print(f"\t{int(split[0] * 100)}% for validation defines each fold.")
            print(f"\tSame {int(split[1] * 100)}% for testing across all folds")

        # Check if CV directories already exist
        CV_path = config['data_path']
        if os.path.exists(CV_path):
            existing_cv_dirs = [d for d in os.listdir(CV_path) if d.startswith('CV') and os.path.isdir(os.path.join(CV_path, d))]
            if len(existing_cv_dirs) == CV_sets:
                print(f"{CV_sets} CV directories already exist at {CV_path}. Skipping CV split.")
                return None
            elif len(existing_cv_dirs) != 0 and len(existing_cv_dirs) != CV_sets:
                print(f"Found {len(existing_cv_dirs)} existing CV directories at {CV_path}. Producing {CV_sets} NEW CV sets instead.")
                if len(existing_cv_dirs) > CV_sets:
                    # print(f"Removing {len(existing_cv_dirs) - CV_sets} existing CV directories.")
                    for d in existing_cv_dirs[CV_sets:]:
                        shutil.rmtree(os.path.join(CV_path, d))
            elif len(existing_cv_dirs) == 0:
                print(f"\nNo existing CV directories found at {CV_path}. Creating new CV sets.")
        else:
            os.makedirs(CV_path)
            # print(f"Created directory for CV sets at {CV_path}")
            print(f"\nProducing {CV_sets} CV sets at {CV_path}.")
        
        # mask as 0 for training, 1 for validation and 2 for test
        # add mask ALSO for edge_types where edge_type[0/1] == 'disease' - This are the MSG-PASS edges used for predicting indications, so we need to split them as well

        for edge_type in data.edge_types:
            # Assign first split[1] % of edges for test set to be shared across all CV sets
                # if edge_type[0] == 'disease':
                    # print(edge_type)
                if not masked:
                    data[edge_type]['mask'] = torch.zeros(data[edge_type].num_edges)

                    if len(split) == 2:
                    # if edge_type[1] == rel and config['global_neg_edges']:
                    #     data[edge_type]['mask'][int(num_indications * (1 - split[1])):num_indications + 1] = 2  # Test set for pos edges

                    #     indices = list(range(num_indications))
                    #     random.shuffle(indices)
                    #     neg_indices = list(range(num_indications, total_edges))
                    #     indices = indices + neg_indices # Ensures no mixing of positive and negative edges while shuffling. Next step, neg edges mask are added

                    # else:
                        indices = list(range(data[edge_type].num_edges))
                        random.shuffle(indices)

                        data[edge_type].edge_index = data[edge_type].edge_index[:, indices]
                        data[edge_type]['mask'][int(len(indices) * (1 - split[1])):] = 2  # Test set

        # Cross-validation sets
        for i in range(CV_sets):
            data_fold = copy.deepcopy(data)

            processed_edge_types = set()
            for edge_type in data_fold.edge_types:
                    # print('edge_type:', edge_type[1])
                # if edge_type[0] == 'disease':

                    # if edge_type[1] == rel and config['global_neg_edges']:
                    #     indices = list(range(num_indications))
                    # else:
                if edge_type[1] in processed_edge_types:
                    continue
                else:
                    processed_edge_types.add(edge_type[1])
                    
                    if masked: # make Outer CV val split into test for inner CV to exclude them from training for inner CV
                        # print('Making inner CV split from already masked data...')
                        edge_mask = data_fold[edge_type]['mask']
                        # print('mask:', edge_mask.shape)
                        # print('mask == 1:', sum(edge_mask == 1))
                        edge_mask[edge_mask == 1] = 2
                        # print('mask:', edge_mask.unique(), '\n')
                        data_fold[edge_type]['mask'] = torch.cat((edge_mask[edge_mask == 0], edge_mask[edge_mask == 2]))
# DOUBLE                        edge_index = data_fold[edge_type].edge_index
# CHECK                        data_fold[edge_type].edge_index = torch.cat((edge_index[:, edge_mask == 0], edge_index[:, edge_mask == 2]))
# outer's test set and the inner cvs' test set do not currently match - Apr 22 DOUBLE CHECK!!
                        indices = list(range(data_fold[edge_type].num_edges - sum(edge_mask == 2)))
                        if edge_type[1] == config['rel']:
                            print('mask == 2:', sum(data_fold[edge_type]['mask'] == 2), f"({(sum(data_fold[edge_type]['mask'] == 2)/data_fold[edge_type].num_edges) * 100:.2f}%)")
                    else:
                        indices = list(range(data_fold[edge_type].num_edges))

                    # print('indices:', len(indices))
                    # print('mask:', data_fold[edge_type]['mask'].shape)
                    # print('edges:', data_fold[edge_type].num_edges)

                    fold_size = int(len(indices) * split_ptg // CV_sets)

                    # Assign valudation mask for CV-split edges
                    val_start = i * fold_size
                    val_end = (i + 1) * fold_size if i < (CV_sets - 1) else int(len(indices) * split_ptg)
                    data_fold[edge_type]['mask'][val_start:val_end] = 1  # Validation set

                    if edge_type[1] == config['rel']:
                        print('fold size:', fold_size)
                        print('mask == 1:', sum(data_fold[edge_type]['mask'] == 1), f"({(sum(data_fold[edge_type]['mask'] == 1)/data_fold[edge_type].num_edges) * 100:.2f}%)")

                    # if edge_type[1] == rel and config['global_neg_edges']:
                    #     # Only split positive edges for CV
                    #     neg_indices = list(range(num_indications, int(len(total_edges) * (1 - split[1]))))
                    #     neg_val = random.sample(neg_indices, int(len(neg_indices) * split[0]))
                    #     data_fold[edge_type]['mask'][neg_val] = 1  # Validation set

                    #     indices = list(range(data_fold[edge_type].num_edges)) # Update indices after adding neg edges val split to shuffle ALL edges

                    indices = list(range(data_fold[edge_type].num_edges))
                    random.shuffle(indices)

                    # print('indices:', len(indices))
                    # print('mask:', data_fold[edge_type]['mask'].shape)
                    # print('edges:', data_fold[edge_type].num_edges)

                    data_fold[edge_type].edge_index = data_fold[edge_type].edge_index[:, indices]
                    data_fold[edge_type].mask = data_fold[edge_type].mask[indices]

                    data_fold[edge_type[2], edge_type[1], edge_type[0]].edge_index = data_fold[edge_type].edge_index.flip(0)
                    data_fold[edge_type[2], edge_type[1], edge_type[0]].mask = data_fold[edge_type]['mask']

            # Save to CV_path directory
            cv_dir = os.path.join(CV_path, f'CV{i+1}')
            os.makedirs(cv_dir, exist_ok=True)
            torch.save(data_fold, os.path.join(cv_dir, 'data.pt'))
            print(f"Saved CV set {i+1} to {cv_dir}")

        return None

def process_data(data, config, split = [.2], cross_validate = True):
    node_type1 = config['node_type1']
    node_type2 = config['node_type2']
    rel = config['rel']
    seeded_type = config['seeded_type']

    # Add the input nodes: nodes to be seeded during sampling
    data[seeded_type]['input_nodes'] = torch.zeros(data[seeded_type].num_nodes, dtype=torch.bool)
    if seeded_type == node_type2:
        seeded_nodes = data[(node_type1, rel, node_type2)]['edge_index'][1].unique()
    elif seeded_type == node_type1:
        seeded_nodes = data[(node_type1, rel, node_type2)]['edge_index'][0].unique()
    else:
        raise ValueError(f"Seeded type {seeded_type} is not valid. Must be either {node_type1} (node_type1) or {node_type2} (node_type2).")
    
    # Set the input nodes for the seeded type
    data[seeded_type]['input_nodes'][seeded_nodes] = True


    return data_split(data, config, split, cross_validate)


def process_batch(batch, config, stage = 'train'): #, ind_MSG_Pass = True):
    node_type1 = config['node_type1']
    node_type2 = config['node_type2']
    rel = config['rel']

    # Extract the edge index and labels
    labeled = 'edge_label' in batch[(node_type1, rel, node_type2)]

    # Use the mask to filter the edge index and labels
    processed_edge_types = set()
    for edge_type in batch.edge_types:
        # if edge_type[0] == 'disease' or edge_type[2] == 'disease':
        if edge_type[1] in processed_edge_types:
            continue
        else:
            processed_edge_types.add(edge_type[1])
            # print(f"Processing edge type: {edge_type}")

            # if edge_type[0] == config['seeded_type']: # and edge_type[1] == rel:
            #     continue

            edge_index = batch[edge_type].edge_index
            edge_mask = batch[edge_type].mask
            # print(f"\tEdge index shape: {edge_index.shape}, Edge mask shape: {edge_mask.shape}")

            if stage == 'train':
                mask = edge_mask == 0
            elif stage == 'val':
                mask = edge_mask != 2
            elif stage == 'test':
                mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
            else:
                raise ValueError(f"Invalid data stage mask type: {stage}. Must be 'train', 'val', or 'test'.")

            edge_index = edge_index[:, mask]
            edge_mask = edge_mask[mask]
            # print(f"Filtered edge index shape: {edge_index.shape}, Filtered edge mask shape: {edge_mask.shape}")

            if edge_type[1] == rel:

                if labeled:
                    edge_label = batch[edge_type].edge_label
                    edge_label = edge_label[mask]

                    # if 0 in edge_label:
                    #     indications = indications[:, edge_label != 0] 

                # if config['ind_MSG_Pass']:
                if config['sampling_layers'] > 0:
                    if config['seeded_type'] == node_type2:
                        batch_index = (edge_index[1] < config['batch_size'])
                    elif config['seeded_type'] == node_type1:
                        batch_index = (edge_index[0] < config['batch_size'])
                else:
                    # batch_index = torch.zeros(edge_index.shape[1], dtype=torch.bool)
                    # idx = random.sample(range(edge_index.shape[1]), config['batch_size'])
                    # batch_index[idx] = True

                    # print(f"Batch idx: {idx}")

                    batch_index = torch.ones(edge_index.shape[1], dtype=torch.bool)

                # print(f"Batch index shape: {batch_index.shape}, sum: {batch_index.sum().item()} ({(batch_index.sum().item()/batch_index.shape[0])*100:.2f}%)")
                # print(f"batch == 0: {(batch_index == 0).sum().item()} out of {batch_index.shape[0]} edges.")
                # print(f"Total edges before splitting: {edge_index.shape[1]}")

                # else:
                #     batch_index = (edge_index[1] <= edge_index[1].max())

                # print(f"Batch index shape: {batch_index.shape}, sum: {batch_index.sum().item()} ({(batch_index.sum().item()/batch_index.shape[0])*100:.2f}%)")

                if config['seeded_ind_only']:
                    edge_index = edge_index[:, batch_index]
                    edge_label = edge_label[batch_index] if labeled else None
                    indications = edge_index
                else:
                    edges_batch = edge_index[:, batch_index]
                    labels_batch = edge_label[batch_index] if labeled else None
                    indications = edges_batch

                    edge_index = edge_index[:, ~batch_index]
                    edge_label = edge_label[~batch_index] if labeled else None
                    # print(f"Initial edge_index shape: {edge_index.shape}, Labels shape: {edge_label.shape if labeled else 'N/A'}")

                # print(f"Total indiciations edges in batch: {indications.shape[1]}")
                # print(f"Total edge_index in batch: {edge_index.shape[1]}")

                if stage == 'train':
                    num_batch = indications.shape[1]
                    target_batch = torch.zeros(num_batch, dtype=torch.bool)
                    targets = random.sample(range(num_batch), int(num_batch*config['sup_split']))
                    target_batch[targets] = True

                    if not config['seeded_ind_only']:
                        num_index = edge_index.shape[1]
                        index_MSGPASS = torch.zeros(num_index, dtype=torch.bool)
                        MSGPASS = random.sample(range(num_index), int(num_index*(1 - config['sup_split'])))
                        index_MSGPASS[MSGPASS] = True

                        edge_index = edge_index[:, index_MSGPASS]
                        edge_label = edge_label[index_MSGPASS] if labeled else None
                        # print(f"After MSG-PASS filtering (MSG-PASS: {index_MSGPASS.sum().item()}), edge_index shape: {edge_index.shape}, Labels shape: {edge_label.shape if labeled else 'N/A'}")

                else: 
                    mask_batch = edge_mask[batch_index]
                    target_batch = mask_batch == 1 if stage == 'val' or 2 not in mask_batch else mask_batch == 2

                # print(f"Target batch sum: {target_batch.sum().item()} out of {target_batch.shape[0]} edges.")

                if config['seeded_ind_only']:
                    edges_batch = edge_index[:, target_batch]
                    labels_batch = edge_label[target_batch] if labeled else None
                    edge_index = edge_index[:, ~target_batch]
                    edge_label = edge_label[~target_batch] if labeled else None

                else:
                    add_to_index = edges_batch[:, ~target_batch]
                    add_to_labels = labels_batch[~target_batch] if labeled else None

                    edges_batch = edges_batch[:, target_batch]
                    labels_batch = labels_batch[target_batch] if labeled else None

                    edge_index = torch.cat((edge_index, add_to_index), dim=1)
                    edge_label = torch.cat((edge_label, add_to_labels)) if labeled else None
                    # print(f"added {add_to_index.shape[1]} edges and labels for MSG-PASS")
                    # print(f"After adding to MSG-PASS, edge_index shape: {edge_index.shape}, Labels shape: {edge_label.shape if labeled else 'N/A'}")

                # print(f"Edges used = {sum(mask)/len(mask) * 100:.2f}% of {mask.shape[0]} edges.") if stage == 'val' else None
                # print(f"Ptg supervision edges in batch {edges_batch.shape[1] / (edges_batch.shape[1] + edge_index.shape[1]) * 100:.2f}% of {edges_batch.shape[1] + edge_index.shape[1]} total edges.")

                if labeled:
                    if 0 in edge_label:
                        edge_index = edge_index[:, edge_label != 0] 
                        edge_label = edge_label[edge_label != 0]

                    # Not really needed, but for consistency
                    batch[edge_type].edge_label = edge_label 
                    batch[(edge_type[2], edge_type[1], edge_type[0])].edge_label = edge_label


                # print(f"edge_index shape: {edge_index.shape}, Labels batch shape: {edge_label.shape if labeled else 'N/A'}")
                # batch[(edge_type[2], edge_type[1], edge_type[0])].edge_index = edge_index.flip(0)

            batch[edge_type].edge_index = edge_index
            batch[(edge_type[2], edge_type[1], edge_type[0])].edge_index = edge_index.flip(0)
            # print(f"Updated edge index shape: {batch[edge_type].edge_index.shape}\n")

    if labeled:
        if 2 in labels_batch and config['balanced']:
            # Make edges_batch balanced btwn contraindications and indications; do not remove from edge_index
            contras = edges_batch[:, labels_batch == 2]
            contras = contras[:,random.sample(range(contras.shape[1]), (labels_batch == 1).sum().item())]

            edges_batch = torch.cat((edges_batch[:, labels_batch != 2], contras), dim=1)
            labels_batch = torch.cat((labels_batch[labels_batch != 2], torch.full((contras.shape[1],), 2)))

        num_negs = (labels_batch == 1).sum().item() if 0 not in labels_batch else None

    elif not labeled:
        labels_batch = torch.ones(edges_batch.shape[1])
        num_negs = edges_batch.shape[1]

    if 0 not in labels_batch:
        # Generate negative edges
        # neg_edges = make_neg_edges(edges_batch, num_negs)
        neg_edges = make_neg_edges(indications, num_negs)
        # could generate an edge that exists but is not present on edges_batch as a neg edge... what's the probably that sampled drug/disease pair will not have a sampled indication? 
        # for validation mask == 1, much likely an existing ind is not in edges_batch

        # print(neg_edges.shape, '\tneg_edges shape:')
        # print(edges_batch.shape, '\tedges_batch shape before')
        # print(edge_label.shape, '\tlabels_batch shape before')
        # print(edge_label.unique(), '\tlabels_batch unique before')

        labels_batch = torch.cat((labels_batch, torch.zeros(neg_edges.shape[1])))
        edges_batch = torch.cat((edges_batch, neg_edges.to(edges_batch.device)), dim=1)

    # print(edges_batch.shape, '\tedges_batch shape after')
    # print(labels_batch.shape, '\tlabels_batch shape after')
    # print(labels_batch.unique(), '\tlabels_batch unique after')

    # Shuffle the indices 
    shuffle_indices = torch.randperm(labels_batch.size(0))
    labels_batch = labels_batch[shuffle_indices]
    edges_batch = edges_batch[:, shuffle_indices]


#    print('batch size (edges):', labels_batch.shape)
#    print('real edges in batch:', batch[(node_type1, rel, node_type2)].edge_index.shape)

    return batch, edges_batch, labels_batch


