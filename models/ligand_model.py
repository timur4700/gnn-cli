from models.mpnn_layers import *
from models import utils
from typing import Literal
from torch.nn import Module, Sequential, ModuleList


class LigandModel(Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 mpnn_type: Literal['gcn', 'gin', 'gat'],
                 edge_attr=True,
                 edge_dim=0,
                 num_mpnn_layers: int=2,
                 act_function='relu',
                 glob_pool: Literal['sum', 'mean', 'max']='sum'):
        
        super().__init__()

        self.node_features_embedd = Sequential(Linear(input_dim, hidden_dim), 
                                   make_act_function(act_function)(), 
                                   Linear(hidden_dim, hidden_dim))

        self.mpnn_modules = {'gcn': GCNLayer,
                            'gin': GINlayer,
                            'gat': GATConvLayer}
        
        self.pool = utils.make_pooling(glob_pool)


        self.mpnn_layers = ModuleList([self.mpnn_modules[mpnn_type](node_input_dim=hidden_dim,
                                                                 node_hidden_dim=hidden_dim, edge_attr=edge_attr, edge_dim=edge_dim) for _ in range(num_mpnn_layers)])

        self.scalar_head = Sequential(Linear(hidden_dim, hidden_dim), 
                                   make_act_function(act_function)(), 
                                   Linear(hidden_dim, hidden_dim),
                                   make_act_function(act_function)(),
                                   Linear(hidden_dim, output_dim))
        


    def forward(self, batch):

        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        x = self.node_features_embedd(x)

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr, batch_idx)

        x = self.pool(x, batch_idx)
        out = self.scalar_head(x)

        return out