from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

# class HGT(nn.Module):
#     def __init__(self, hidden_channels, out_channels, num_heads, num_layers,
#                  node_types, metadata, config):
#         super().__init__()

#         self.node_type1 = config['node_type1']
#         self.node_type2 = config['node_type2']

#         self.lin_dict = nn.ModuleDict()
#         for node_type in node_types: # data.node_types:
#             self.lin_dict[node_type] = Linear(-1, hidden_channels[0])
            
#         self.convs = nn.ModuleList()
#         for i in range(num_layers):
#             conv = HGTConv(hidden_channels[i], hidden_channels[i+1], metadata,
#                            num_heads[i])#, group='mean') # data.metadata()
#             self.convs.append(conv)
        
#         self.lin = Linear(sum(hidden_channels[1:]), out_channels)
        
#         self.dropout = nn.Dropout(config['dropout'])

#     def forward(self, x_dict, edge_index_dict):
#         x_dict = {
#             node_type: self.dropout(self.lin_dict[node_type](x).relu_())
#             for node_type, x in x_dict.items()
#         }
#         out = {}
#         for i, conv in enumerate(self.convs):
#             x_dict = conv(x_dict, edge_index_dict)

#             if out=={}:
#                 out = copy.copy(x_dict)
#             else:
#                 out = {
#                     node_type: torch.cat((out[node_type], x_dict[node_type]), dim=1)
#                     for node_type, x in x_dict.items()
#                 }

#         return F.relu(self.lin(out[self.node_type1])), F.relu(self.lin(out[self.node_type2]))


class HGT(nn.Module):
    def __init__(self, hidden_channels, #out_channels, num_heads, num_layers,
                 node_types, metadata, config):
        super().__init__()

        self.node_type1 = config['node_type1']
        self.node_type2 = config['node_type2']
        self.residuals = config['Residuals']

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels[0])

        self.convs = nn.ModuleList()
        if self.residuals:
            self.res_proj = nn.ModuleList()
        for i in range(config['GNN_layers']):
            conv = HGTConv(hidden_channels[i], hidden_channels[i+1], metadata, config['att_heads'][i]) #, group='mean')
            self.convs.append(conv)
            if self.residuals:
                proj_dict = nn.ModuleDict()
                for node_type in node_types:
                    if hidden_channels[i] != hidden_channels[i+1]:
                        proj_dict[node_type] = Linear(hidden_channels[i], hidden_channels[i+1])
                    else:
                        proj_dict[node_type] = nn.Identity()
                self.res_proj.append(proj_dict)

        self.lin = Linear(sum(hidden_channels[1:]), config['out_channel'])
        self.dropout = nn.Dropout(config['GNN_dropout'])

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.dropout(self.lin_dict[node_type](x).relu_())
            for node_type, x in x_dict.items()
        }
        out = {}
        for i, conv in enumerate(self.convs):
            x_dict_in = x_dict
            x_dict = conv(x_dict, edge_index_dict)
            if self.residuals:
                x_dict = {
                    node_type: x_dict[node_type] + self.res_proj[i][node_type](x_dict_in[node_type])
                    for node_type in x_dict
                }

            if out == {}:
                out = copy.copy(x_dict)
            else:
                out = {
                    node_type: torch.cat((out[node_type], x_dict[node_type]), dim=1)
                    for node_type, x in x_dict.items()
                }

        return F.relu(self.lin(out[self.node_type1])), F.relu(self.lin(out[self.node_type2]))

#

class MLPPredictor(nn.Module):
    def __init__(self, channel_num, dropout, class_num = 1):
        super().__init__()
        self.L1 = nn.Linear(channel_num * 2, channel_num)
        self.L2 = nn.Linear(channel_num, class_num)
        self.bn = nn.BatchNorm1d(num_features=channel_num)
        self.dropout = nn.Dropout(dropout)
        self.class_num = class_num

    def forward(self, drug_embeddings, disease_embeddings):
        x = torch.cat((drug_embeddings, disease_embeddings), dim=1)
        x = F.relu(self.bn(self.L1(x)))
        x = self.dropout(x)
        x = self.L2(x)
        return x

def compute_loss(scores, labels):
    """
    Computes cross-entropy loss for binary and multi-class classification.
    Assumes `scores` are raw logits of shape (batch_size, num_classes) for multi-class
    or (batch_size, 1) for binary classification,
    and `labels` are integer class indices of shape (batch_size,).
    """
    if scores.shape[1] == 1:
        pos_weights = torch.clone(labels)
        pos_weights[pos_weights == 1] = ((labels==0).sum() / labels.shape[0])
        pos_weights[pos_weights == 0] = ((labels==1).sum() / labels.shape[0])

        return F.binary_cross_entropy_with_logits(scores[:, 0], labels, pos_weight=pos_weights)
        #     return F.binary_cross_entropy_with_logits(scores, labels)
    else:
        num_classes = scores.shape[1]
        class_counts = torch.bincount(labels.long(), minlength=num_classes)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * num_classes  # normalize to sum to num_classes, optional

        return F.cross_entropy(scores, labels.long(), weight=class_weights)



#  