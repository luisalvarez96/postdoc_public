import pandas as pd
import torch
from torch_geometric.data import HeteroData
import argparse
import os
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../data')
    parser.add_argument('--filter', type=str, nargs='*', default=['anatomy', 'path', 'molfunc', 'biop', 'cellcomp', 'exp'],
                        help='Relations to filter from KG by keywords e.g., ' \
                        'anatomy for anatomy_*, path for pathway_*, molfunc for mol_function_*, biop for bioprocess_*, cellcomp for cellcomp_*, exp for exposure_*'
                        )
    parser.add_argument('--training_kg', action='store_true', help='Filter KG to only include drugs and diseases with indications')
    parser.add_argument('--no-training_kg', dest='training_kg', action='store_false', help='Do not filter KG')
    parser.set_defaults(training_kg=True)
    parser.add_argument('--node_type1', type=str, default='drug')
    parser.add_argument('--node_type2', type=str, default='disease')
    parser.add_argument('--rel', type=str, default='indication')
    parser.add_argument('--merge_label', action='store_true', help='Merge indication and contraindication labels')
    parser.add_argument('--no-merge_label', dest='merge_label', action='store_false', help='Do not merge labels')
    parser.set_defaults(merge_label=True)
    return parser.parse_args()

def genes_into_edges(nipah_genes, KG, nipah_index = "32352"):
    nipah_genes.columns = ['gene']
    nipah_genes['gene'] = nipah_genes['gene'].str.replace(' ', '')
    nipah_genes = nipah_genes.drop_duplicates()
    nipah_genes['KG_index'] = pd.NA
    nipah_genes.reset_index(drop=True, inplace=True)

    for i in range(len(nipah_genes)):
        index = KG[KG['x_name'] == nipah_genes['gene'].iloc[i]]['x_index'].unique()
        if len(index) == 1:
            nipah_genes.loc[i, "KG_index"] = 'gene/protein::' + ','.join(index.astype(str))
        elif len(index) == 0:
            index = KG[KG['y_name'] == nipah_genes['gene'].iloc[i]]['y_index'].unique()
            if len(index) == 1:
                nipah_genes.loc[i, "KG_index"] = 'gene/protein::' + ','.join(index.astype(str))
            elif len(index) == 0:
                print(i, nipah_genes['gene'].iloc[i], 'not in KG')
            else:
                print(i, nipah_genes['gene'].iloc[i], 'multiple index in KG')
        else:
            print(i, nipah_genes['gene'].iloc[i], 'multiple index in KG')

    nipah_genes = nipah_genes.dropna()
    nipah_genes = nipah_genes.reset_index(drop=True)

    nipah_index = KG[KG['x_name'].str.contains('Nipah', regex=True, na=False)]['x_index'].unique()

    nipah_gene_edges = pd.DataFrame({
        0: nipah_genes['KG_index'],
        1: 'disease_protein',
        2: 'disease::' + ','.join(nipah_index.astype(str))
        })
    nipah_gene_edges = pd.concat([nipah_gene_edges, 
        pd.DataFrame({
        0: nipah_gene_edges[2],
        1: nipah_gene_edges[1],
        2: nipah_gene_edges[0]
    })], ignore_index=True)

    return nipah_gene_edges

def insert_entry(entry, ent_type, dic):
    if ent_type not in dic:
        dic[ent_type] = {}
    ent_n_id = len(dic[ent_type])
    if entry not in dic[ent_type]:
         dic[ent_type][entry] = ent_n_id
    return dic

def KG_into_HeteroData(triplets, entity_dictionary):
    
    edge_dictionary={}
    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]
        
        src_int_id = entity_dictionary[src_type][src]
        dest_int_id = entity_dictionary[dest_type][dest]
        
        pair = (src_int_id,dest_int_id)
        etype = (src_type, triple[1],dest_type)
        if etype in edge_dictionary:
            edge_dictionary[etype] += [pair]
        else:
            edge_dictionary[etype] = [pair]


    print('making hetero graph data object')
    data = HeteroData()
    for i, key in enumerate(entity_dictionary.keys()):
        data[key].x = (torch.ones((len(entity_dictionary[key]), 768)) * i)
        data[key].id = torch.arange(len(entity_dictionary[key]))

    for key in edge_dictionary:
        data[key].edge_index = torch.transpose(torch.IntTensor(edge_dictionary[key]), 0, 1).long().contiguous()


    print('adding embeddings')
    embeddings = pd.read_pickle('../data/entities_embeddings.pkl')
    smiles_embeddings = pd.read_pickle('../data/smiles_embeddings.pkl')

    for i, row in smiles_embeddings.iterrows():
        if row['id'] in entity_dictionary['drug'].keys():
            drug_id = entity_dictionary['drug'][row['id']]
            data['drug'].x[drug_id][:767] = torch.Tensor(row['embedding'])

    for i in range(data['drug'].x.shape[0]):
        data['drug'].x[i, -1] = data['drug'].x[i, :-1].mean()

    for i, row in embeddings.iterrows():
        x_type = row['id'].split('::')[0]
        if x_type in data.node_types and row['id'] in entity_dictionary[x_type] and x_type != 'drug':
            id_ = entity_dictionary[x_type][row['id']]
            data[x_type].x[id_] = torch.Tensor(row['embedding']) 

    return data

def training_KG(KG, node_type1 = 'drug', node_type2 = 'disease', rel = 'indication'):
    print(f'\nfiltering KG to only include drugs and diseases with {rel} edges')
    
    drug_disease_pairs = KG[KG['relation']==rel]
    drugs, diseases = [], []

    for i, row in drug_disease_pairs.iterrows():
        if row['x_type'] == node_type1:
            drugs.append(row['x_index'])
        if row['x_type'] == node_type2:
            diseases.append(row['x_index'])
        
        if row['y_type'] == node_type1:
            drugs.append(row['y_index'])
        if row['y_type'] == node_type2:
            diseases.append(row['y_index'])
            
    drugs, diseases = list(set(drugs)), list(set(diseases))
    to_remove = KG[KG['x_type']==node_type1]
    to_remove = to_remove[~to_remove['x_index'].isin(drugs)]
    KG.drop(to_remove.index, inplace = True)
    to_remove = KG[KG['y_type']==node_type1]
    to_remove = to_remove[~to_remove['y_index'].isin(drugs)]
    KG.drop(to_remove.index, inplace = True)
    to_remove = KG[KG['x_type']==node_type2]
    to_remove = to_remove[~to_remove['x_index'].isin(diseases)]
    KG.drop(to_remove.index, inplace = True)
    to_remove = KG[KG['y_type']==node_type2]
    to_remove = to_remove[~to_remove['y_index'].isin(diseases)]
    KG.drop(to_remove.index, inplace = True)

    return KG

def filter_KG(KG, rels_to_remove = []):
    print(f'\nfiltering KG to remove {rels_to_remove} relations')
    to_remove = pd.DataFrame()
    for keyword in rels_to_remove:
        print(f'for keyword {keyword}: removing {len(KG[KG["relation"].str.contains(keyword, regex=True, na=False)])} edges')
        to_remove = pd.concat([to_remove, KG[KG["relation"].str.contains(keyword)]])#, regex=True, na=False)]], ignore_index=True)
    KG.drop(to_remove.index, inplace = True)
    return KG

def merge_contras(data, node_type1 = 'drug', node_type2 = 'disease', rel = 'indication'):
    # Get edge indices for 'indication' and 'contraindication'
    indication_edges = data[(node_type1, 'indication', node_type2)].edge_index
    contra_edges = data[(node_type1, 'contraindication', node_type2)].edge_index

    # Create edge labels: 1 for indication, 0 for contraindication
    indication_labels = torch.ones(indication_edges.size(1), dtype=torch.long)
    contra_labels = torch.full((contra_edges.size(1),), 2, dtype=torch.long)

    # Concatenate edge indices and labels
    merged_edge_index = torch.cat([indication_edges, contra_edges], dim=1)
    merged_edge_label = torch.cat([indication_labels, contra_labels], dim=0)

    # Assign to the 'indication' relation
    data[(node_type1, rel, node_type2)].edge_index = merged_edge_index
    data[(node_type1, rel, node_type2)].edge_label = merged_edge_label

    data[(node_type2, rel, node_type1)].edge_index = merged_edge_index.flip(0) # CHECK THAT THIS WORKS FINE
    data[(node_type2, rel, node_type1)].edge_label = merged_edge_label

    # Remove the 'contraindication' edge type from the data object
    del data[(node_type1, 'contraindication', node_type2)]
    del data[(node_type2, 'contraindication', node_type1)]

    print(f"Edge types merged: {indication_edges.size(1)} indication edges and {contra_edges.size(1)} contraindication edges.")

    return data

def pred_graphs(nipah_GeneSet_path, KG, KG_parsed, entity_dictionary, save_path, merge = False):
    GeneSet_name = Path(nipah_GeneSet_path).stem.split('_')[0]
    print(f'reading nipah associated genes in {GeneSet_name}')

    GeneSet_pd = pd.read_csv(nipah_GeneSet_path, header=None)

    GeneSet_edges = genes_into_edges(GeneSet_pd, KG)

    print(f'\nGenerating graph for {GeneSet_name} genes')
    KG_parsed_gene_set = pd.concat([KG_parsed, GeneSet_edges], ignore_index=True)
    triplets_gene_set = KG_parsed_gene_set.values.tolist()
    data_gene_set = KG_into_HeteroData(triplets_gene_set, entity_dictionary)

    if merge:# args.merge_label:
        print(f'\nMerging indication and contraindication labels for {GeneSet_name} graph')
        data_gene_set = merge_contras(data_gene_set)

    torch.save(data_gene_set, save_path + f'/hetero_graph_{GeneSet_name}.pt')
    print(f'\nGraph for {GeneSet_name} genes saved to {save_path}/hetero_graph_{GeneSet_name}.pt')

    return data_gene_set

if __name__ == '__main__':
    args = parse_args()
    node_type1 = args.node_type1
    node_type2 = args.node_type2
    rel = args.rel
    save_path = args.save_path
    primekg_file = args.data_directory + '/kg.csv'
    nipah_path = args.data_directory + '/ForNipah'
    rels_to_remove = args.filter
    # filter_kg = args.filter
    training_kg = args.training_kg

    start_time = time.time()

    if os.path.exists(save_path + '/hetero_graph.pt'):
        response = input(f'File already exists at {save_path}/hetero_graph.pt. Overwrite? (y/n): ').strip().lower()
        if response != 'y':
            print('Halting the script.')
            exit(0)

    print('\nreading KG in', primekg_file)
    #primekg_file = '../data/kg.csv'
    KG = pd.read_csv(primekg_file, sep =",")
    print('\nKG head:\n', KG.head())

    print('\nKG breakdown by % of relation types:')
    print(KG['relation'].value_counts() / len(KG) * 100)

    if rels_to_remove:
        print(f'\nRelations to remove from KG: {rels_to_remove}')
        KG = filter_KG(KG, rels_to_remove)

        print('\nKG breakdown by % of relation types (after removing relations):')
        print(KG['relation'].value_counts() / len(KG) * 100)

    nipah_index = KG[KG['x_name'].str.contains('Nipah', regex=True, na=False)]['x_index'].unique()
    print('\nNipah index:', nipah_index)

    print('\ngenerating edges list')
    KG_parsed = pd.DataFrame({
        0: KG['x_type'] + '::' + KG['x_index'].astype(str),
        1: KG['relation'],
        2: KG['y_type'] + '::' + KG['y_index'].astype(str)
    })
    KG_parsed = KG_parsed.drop_duplicates()

    triplets = KG_parsed.values.tolist()

    print('\ngenerating entity & edge dictionaries')

    # Only one entity dictionary for all triplets so that all graphs use same entity ids, when running the prediction script, the same entity ids correspond to the same entities (drugs)
    entity_dictionary = {}
    for triple in triplets:
        src = triple[0]
        split_src = src.split('::')
        src_type = split_src[0]
        dest = triple[2]
        split_dest = dest.split('::')
        dest_type = split_dest[0]
        insert_entry(src,src_type,entity_dictionary)
        insert_entry(dest,dest_type,entity_dictionary)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pd.to_pickle(entity_dictionary, save_path + '/entity_dictionary.pkl')
    print('\nEntity dictionary saved to', save_path + '/entity_dictionary.pkl')

    # Filter KG to only include drugs and diseases with indications
    if training_kg:
        KG = training_KG(KG)

        # Use filtered KG to generate training data
        KG_parsed_filtered = pd.DataFrame({
            0: KG['x_type'] + '::' + KG['x_index'].astype(str),
            1: KG['relation'],
            2: KG['y_type'] + '::' + KG['y_index'].astype(str)
        })
        KG_parsed_filtered = KG_parsed_filtered.drop_duplicates()
        triplets_filtered = KG_parsed_filtered.values.tolist()
        data = KG_into_HeteroData(triplets_filtered, entity_dictionary)

    else:
        data = KG_into_HeteroData(triplets, entity_dictionary)

    if args.merge_label:
        print('\nMerging indication and contraindication labels')
        data = merge_contras(data)

    torch.save(data, save_path + '/hetero_graph.pt')
    print(f'\nTraining graph saved to {save_path}/hetero_graph.pt')

    # Generate separate graphs for each gene set
    print(f'\nreading nipah associated gene files in {nipah_path}')
    for file in os.listdir(nipah_path):
        if file.endswith('.txt'):
            print(f'\nProcessing gene set file: {file}')
            data_gene_set = pred_graphs(os.path.join(nipah_path, file), KG, KG_parsed, entity_dictionary, save_path, args.merge_label)


    print("\nNode types and their embeddings in training data:")
    for node_type in data.node_types:
        print(data[node_type].x.shape, node_type)

    print('\nEdges in training data:')
    for edge_type in data.edge_types:
        print(f"Number of edges for {edge_type}: {data[edge_type].num_edges}")
    print("Total number of edges:", data.num_edges)

    print(f"\nNode types and their embeddings in data for predictions (from {file}):")
    for node_type in data_gene_set.node_types:
        print(data_gene_set[node_type].x.shape, node_type)

    print(f'\nEdges in data for predictions (from {file}):')
    for edge_type in data_gene_set.edge_types:
        print(f"Number of edges for {edge_type}: {data_gene_set[edge_type].num_edges}")    
    print("Total number of edges:", data_gene_set.num_edges)
    
    if not args.merge_label:
        contra = 'contra' + rel
        print(f'\nIn training data:\n\
            {data[(node_type1, rel, node_type2)].num_edges}\tindications\
            {data[(node_type1, contra, node_type2)].num_edges}\tcontraindications')

        print(f'In data for predictions (from {file}):\n\
            {data_gene_set[(node_type1, rel, node_type2)].num_edges}\tindications\
            {data_gene_set[(node_type1, contra, node_type2)].num_edges}\tcontraindications')
    
    elapsed_time = time.time() - start_time
    print(f'\nPreprocessing completed in {elapsed_time / 60:.2f} minutes.')