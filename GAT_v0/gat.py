import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    """
    Graph Attention Network layer implementation as per:
    https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (N, in_features)
        Wh = torch.mm(h, self.W)  # (N, out_features)
        N = Wh.size()[0]
        
        # Calculate attention coefficients
        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N, N)
        
        # Mask out non-neighbors
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to transformed features
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
