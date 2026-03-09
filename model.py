# coding=utf-8


import torch.nn as nn
import torch
import torch.nn.functional as F


class ComHyperConvLayer(nn.Module):
    

    def __init__(self, emb_dim, device):
        super(ComHyperConvLayer, self).__init__()

        # self.fc_seq = nn.Linear(2 * emb_dim, emb_dim, bias=True, device=device)
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, items_embs, pad_all_train_outfits, HG_up, HG_pu):


        # 1. node -> hyperedge message
        # 1) item node aggregation
        msg_item_agg = torch.sparse.mm(HG_up, items_embs)  # [U, d]

        # 2. propagation: hyperedge -> node
        # propag_items_embs = torch.sparse.mm(HG_item_outfit, msg_emb)    # [L, d]
        propag_items_embs = torch.sparse.mm(HG_pu, msg_item_agg)  # [L, d]
        # propag_items_embs = self.dropout(propag_items_embs)

        return propag_items_embs


class PreHyperConvLayer(nn.Module):


    def __init__(self):
        super(PreHyperConvLayer, self).__init__()

    def forward(self, items_embs, HG_item_user, HG_item_outfit):
        msg_tar = torch.sparse.mm(HG_item_outfit, items_embs)
        msg_src = torch.sparse.mm(HG_item_user, msg_tar)

        return msg_src


class ComHyperConvNetwork(nn.Module):
   


    def __init__(self, num_layers, emb_dim, dropout, device):
        super(ComHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.user_hconv_layer = ComHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, items_embs, pad_all_train_outfits, HG_up, HG_pu):
        final_items_embs = [items_embs]
        for layer_idx in range(self.num_layers):
            items_embs = self.user_hconv_layer(items_embs, pad_all_train_outfits, HG_up, HG_pu)  # [L, d]
            # add residual connection to alleviate over-smoothing issue
            items_embs = items_embs + final_items_embs[-1]
            items_embs = F.dropout(items_embs, self.dropout)
            final_items_embs.append(items_embs)
        final_items_embs = torch.mean(torch.stack(final_items_embs), dim=0)  # [L, d]

        return final_items_embs


class PreHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        super(PreHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = PreHyperConvLayer()

    def forward(self, items_embs, HG_item_user, HG_item_outfit):
        final_items_embs = [items_embs]
        for layer_idx in range(self.num_layers):
            items_embs = self.di_hconv_layer(items_embs, HG_item_user, HG_item_outfit)
            # add residual connection
            items_embs = items_embs + final_items_embs[-1]
            items_embs = F.dropout(items_embs, self.dropout)
            final_items_embs.append(items_embs)
        final_items_embs = torch.mean(torch.stack(final_items_embs), dim=0)  # [L, d]

        return final_items_embs


class PreconvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(PreconvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, items_embs, outfit_graph):
        final_items_embs = [items_embs]
        for _ in range(self.num_layers):
            # items_embs = outfit_graph @ items_embs
            items_embs = torch.sparse.mm(outfit_graph, items_embs)
            items_embs = items_embs + final_items_embs[-1]
            # items_embs = F.dropout(items_embs, self.dropout)
            final_items_embs.append(items_embs)
        output_items_embs = torch.mean(torch.stack(final_items_embs), dim=0)  # [L, d]

        return output_items_embs


class PCM(nn.Module):
    def __init__(self, num_users, num_items, args, device):
        super(PCM, self).__init__()

        # definition
        self.num_users = num_users
        self.num_items = num_items
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        # embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items + 1, self.emb_dim, padding_idx=num_items)

    
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


        self.user_hconv_network = ComHyperConvNetwork(args.num_user_layers, args.emb_dim, 0, device)
        self.outfit_conv_network = PreconvNetwork(args.num_outfit_layers, args.dropout)
        self.di_hconv_network = PreHyperConvNetwork(args.disen, device, args.dropout)

    
        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        
        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

      
        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.w_gate_outfit = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_outfit = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_outfit.data)
        nn.init.xavier_normal_(self.b_gate_outfit.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

    
        self.dropout = nn.Dropout(args.dropout)

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_items(self, hg_items_embs, outfit_items_embs, trans_items_embs):


        # normalization
        norm_hg_items_embs = F.normalize(hg_items_embs, p=2, dim=1)
        norm_outfit_items_embs = F.normalize(outfit_items_embs, p=2, dim=1)
        norm_trans_items_embs = F.normalize(trans_items_embs, p=2, dim=1)

        # calculate loss
        loss_cl_items = 0.0
        loss_cl_items += self.cal_loss_infonce(norm_hg_items_embs, norm_outfit_items_embs)
        loss_cl_items += self.cal_loss_infonce(norm_hg_items_embs, norm_trans_items_embs)
        loss_cl_items += self.cal_loss_infonce(norm_outfit_items_embs, norm_trans_items_embs)

        return loss_cl_items

    def cal_loss_cl_users(self, hg_batch_users_embs, outfit_batch_users_embs, trans_batch_users_embs):
        # normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_outfit_batch_users_embs = F.normalize(outfit_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # calculate loss
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_outfit_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_outfit_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def forward(self, dataset, batch):

        outfit_gate_items_embs = torch.multiply(self.item_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.item_embedding.weight[:-1],
                                                                       self.w_gate_outfit) + self.b_gate_outfit))
        seq_gate_items_embs = torch.multiply(self.item_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.item_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_items_embs = torch.multiply(self.item_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.item_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        # multi-view hypergraph convolutional network
        hg_items_embs = self.user_hconv_network(col_gate_items_embs, dataset.pad_all_train_outfits, dataset.HG_up, dataset.HG_pu)
        # hypergraph structure aware users embeddings
        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_items_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]  # [BS, d]


        outfit_items_embs = self.outfit_conv_network(outfit_gate_items_embs, dataset.item_graph)  # [L, d]
    
        outfit_structural_users_embs = torch.sparse.mm(dataset.HG_up, outfit_items_embs)
        outfit_batch_users_embs = outfit_structural_users_embs[batch["user_idx"]]  # [BS, d]


        trans_items_embs = self.di_hconv_network(seq_gate_items_embs, dataset.HG_item_user, dataset.HG_item_outfit)

        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_items_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]  # [BS, d]

    
        loss_cl_item = self.cal_loss_cl_items(hg_items_embs, outfit_items_embs, trans_items_embs)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, outfit_batch_users_embs, trans_batch_users_embs)

        # normalization
        norm_hg_items_embs = F.normalize(hg_items_embs, p=2, dim=1)
        norm_outfit_items_embs = F.normalize(outfit_items_embs, p=2, dim=1)
        norm_trans_items_embs = F.normalize(trans_items_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_outfit_batch_users_embs = F.normalize(outfit_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

     
        hyper_coef = self.hyper_gate(norm_hg_batch_users_embs)
        outfit_coef = self.gcn_gate(norm_outfit_batch_users_embs)
        trans_coef = self.trans_gate(norm_trans_batch_users_embs)

    
        fusion_batch_users_embs = hyper_coef * norm_hg_batch_users_embs + outfit_coef * norm_outfit_batch_users_embs + trans_coef * norm_trans_batch_users_embs
        fusion_items_embs = norm_hg_items_embs + norm_outfit_items_embs + norm_trans_items_embs

        # prediction
        prediction = fusion_batch_users_embs @ fusion_items_embs.T

        return prediction, loss_cl_user, loss_cl_item



