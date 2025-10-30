import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class HeteroGraphSage(nn.Module):
    def __init__(
        self, 
        n_layers, 
        n_inp_dict, 
        n_hid,
        n_out,
        rel_names,
        dropout,
        feat_drop = 0.05,
        use_batch_norm=True
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm  
        self.layers = nn.ModuleList()
        self.adapt_ws = nn.ModuleDict()
        self.bn_adapt = nn.ModuleDict() if use_batch_norm else None
        self.bn_layers = nn.ModuleList() if use_batch_norm else None


        for ntype, inp_dim in n_inp_dict.items():
            self.adapt_ws[ntype] = nn.Linear(inp_dim, n_hid)
            if use_batch_norm:
                self.bn_adapt[ntype] = nn.LayerNorm(n_hid)

        for _ in range(n_layers):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    { rel: dglnn.SAGEConv(n_hid, n_hid, 'mean', feat_drop=feat_drop) for rel in rel_names}, aggregate='sum')

            )
            if use_batch_norm:
                self.bn_layers.append(nn.LayerNorm(n_hid))
            
        self.out = nn.Linear(n_hid, n_out)
        

    def forward(self, graph, feat, out_key='runtime', eweight=None):
        with graph.local_scope():
            h = {}
            for ntype in graph.ntypes:
                h[ntype] = self.adapt_ws[ntype](feat[ntype])
                if self.use_batch_norm and h[ntype].size(0):
                    h[ntype] = self.bn_adapt[ntype](h[ntype])
                h[ntype] = F.gelu(h[ntype])

            for i in range(self.n_layers):
                if eweight:
                    h = self.layers[i](graph, h, mod_kwargs={rel: {'edge_weight': eweight[rel]} for rel in eweight})
                else:
                    h = self.layers[i](graph, h)
                if self.use_batch_norm:
                    h = {k: v.flatten(1) for k, v in h.items()}
                    h = {k: self.bn_layers[i](v) if v.size(0) > 1 else v for k, v in h.items()}
                
                h = {k: F.gelu(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
        
            if out_key == 'runtime':
                return self.out(h['runtime'])
            else:
                return h


    def predict(self, embeddings, out_key='runtime'):
        return self.out(embeddings[out_key])
