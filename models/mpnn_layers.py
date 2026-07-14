from torch.nn import Module, Linear, Sequential, Dropout  
from models.utils import make_normalization,make_act_function, DropoutEdge
from torch_geometric.nn import GraphNorm, GCNConv, GINConv, GINEConv, GATConv, GATv2Conv



class ModuleMPNN(Module):
    def __init__(self,
                 node_hidden_dim: int,
                 edge_attr=False,
                 edge_dim=None,
                 node_feature_dropout_p: float=0.1,
                 edge_dropout_p: float=0.0,
                 normaliztion: str='batchnorm',
                 residual=True,
                 undirected=True
                 ):
        super().__init__()


        self.gnn_conv_module=None
        self.node_hidden_dim = node_hidden_dim

        self.edge_attr = edge_attr
        self.edge_dim = edge_dim

        self.norm = make_normalization(normaliztion)(self.node_hidden_dim)
        self.dropout_feature = Dropout(node_feature_dropout_p)
        self.dropout_edge = DropoutEdge(edge_dropout_p, undirected=undirected)
        self.residual = residual


    def forward(self, x, edge_index, edge_attr=None, batch_idx=None):

        residual = x 

        edge_index, edge_attr_mask = self.dropout_edge(edge_index)
        edge_attr = edge_attr[edge_attr_mask] if self.edge_attr else None

        x = self.gnn_conv_module(x, edge_index, edge_attr) if self.edge_attr else self.gnn_conv_module(x, edge_index)
        x = self.norm(x, batch_idx) if isinstance(self.norm, GraphNorm) else self.norm(x)
        x = self.dropout_feature(x)
 
        if self.residual:
            x = residual + x
            return x
        
        return x



class GCNLayer(ModuleMPNN):
    def __init__(self,
                 node_input_dim: int,
                 node_hidden_dim: int,
                 improved=False,
                 cached=False,
                 add_self_loops=None,
                 normalize_in=True,
                 bias=True,
                 **kwrags):
        super().__init__(node_hidden_dim=node_hidden_dim, **kwrags)

        self.edge_attr = False
        self.gnn_conv_module = GCNConv(in_channels=node_input_dim,
                                       out_channels=node_hidden_dim,
                                       improved=improved,
                                       cached=cached,
                                       add_self_loops=add_self_loops,
                                       normalize=normalize_in,
                                       bias=bias)



class GINlayer(ModuleMPNN):
    def __init__(self, 
                 node_input_dim: int,
                 node_hidden_dim: int,
                 act_function='relu',
                 eps: int|float=0,
                 train_eps=True, **kwargs):
        
        super().__init__(node_hidden_dim=node_hidden_dim, **kwargs)

        self.fc_module = Sequential(Linear(node_input_dim, node_input_dim), 
                                    make_act_function(act_function)(),
                                    Linear(node_hidden_dim, node_hidden_dim))

        if self.edge_attr:

            assert self.edge_dim, 'Error, edge ffeature input dimension is needed'
            self.gnn_conv_module = GINEConv(nn=self.fc_module,
                                     eps=eps,
                                     edge_dim=self.edge_dim)
            
        else:
            self.gnn_conv_module = GINConv(nn=self.fc_module,
                                    eps=eps,
                                    train_eps=train_eps)



class GATConvLayer(ModuleMPNN):
    def __init__(self,
                 node_input_dim: int,
                 node_hidden_dim: int,
                 gat_version: int=1,
                 num_heads: int=1,
                 concat=True,
                 negative_slope: float=0.2,
                 add_self_loops=True,
                 bias=True,
                 **kwargs):
        super().__init__(node_hidden_dim=node_hidden_dim, **kwargs)

        if self.edge_attr:
            assert self.edge_dim, 'Error, edge feature input dimension is needed'

        if gat_version == 1:
            self.gnn_conv_module = GATConv(in_channels=node_input_dim,
                                        out_channels=node_hidden_dim,
                                        heads=num_heads,
                                        concat=concat,
                                        edge_dim=self.edge_dim if self.edge_attr else None,
                                        add_self_loops=add_self_loops,
                                        negative_slope=negative_slope,
                                        bias=bias)
            
        elif gat_version == 2:
             self.gnn_conv_module = GATv2Conv(in_channels=node_input_dim,
                                        out_channels=node_hidden_dim,
                                        heads=num_heads,
                                        concat=concat,
                                        edge_dim=self.edge_dim if self.edge_attr else None,
                                        add_self_loops=add_self_loops,
                                        negative_slope=negative_slope,
                                        bias=bias)    

        else:
            raise ValueError('Only GAT versions 1 and 2 are available')

