import torch
import torch.nn as nn

from data_generator import *
from EGNN_model import *
from GraphTransformer_Block import *


class FinalModel(nn.Module):
    def __init__(self,input_size,hidden_size,fliter_size,output_size,dropout_rate,n_layers):
        super(FinalModel, self).__init__()
        self.Egnn = EGNN(in_node_nf=input_size, hidden_nf=hidden_size, out_node_nf=output_size, in_edge_nf=2, n_layers=10, attention=True,residual=False)
        self.GT = GraghTransformer(in_channels=input_size,edge_features=2,dropout_rate=dropout_rate,num_layers=4,transformer_residual=False)
    
        self.criterion = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)

    def forward(self, node_features, xyz_feats, edges, edge_att, edge_feat, adj):
        x1 = self.Egnn(node_features, xyz_feats, edges, edge_feat)
        x2 = self.GT(node_features, edge_feat, edges)
        x = (x1 + x2) / 2
        return x
