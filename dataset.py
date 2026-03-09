# coding=utf-8


from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ITEMDataset(Dataset):
    def __init__(self, data_filename, outfit_filename, num_users, num_items, padding_idx, args, device):

        # get all outfits
        self.data = load_list_with_pkl(data_filename)  
        self.outfits_dict = self.data[0]  
        self.labels_dict = self.data[1]
        self.items_dict = load_dict_from_pkl(outfit_filename)

        # definition
        self.num_users = num_users
        self.num_items = num_items
        # self.num_outfits = get_num_outfits(self.outfits_dict)
        self.padding_idx = padding_idx
        self.attr = args.attr
        self.keep_rate = args.keep_rate
        self.device = device

        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.outfits_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

     
        self.item_adj = gen_item_outfit_adj(num_items, self.items_dict, self.attr)   # csr_matrix
        self.item_graph_matrix = normalized_adj(adj=self.item_adj, is_symmetric=False)
        self.item_graph = transform_csr_matrix_to_tensor(self.item_graph_matrix).to(device)

        # generate item-outfit incidence matrix, its degree and hypergraph
        self.H_pu = gen_sparse_H_user(self.outfits_dict, num_items, self.num_users)    # [L, U]
        self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)
        # get degree of H_pu
        self.Deg_H_pu = get_hyper_deg(self.H_pu)    # [L, L]
        # normalize item-user hypergraph
        self.HG_pu = self.Deg_H_pu * self.H_pu    # [L, U]
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        # generate outfit-item incidence matrix, its degree and hypergraph
        self.H_up = self.H_pu.T    # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)    # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up    # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

    
     
        self.all_train_outfits = get_all_users_seqs(self.users_trajs_dict)
    
        self.pad_all_train_outfits = pad_sequence(self.all_train_outfits, batch_first=True, padding_value=padding_idx)
        self.pad_all_train_outfits = self.pad_all_train_outfits.to(device)    # [U, MAX_SEQ_LEN]
        self.max_outfit_len = self.pad_all_train_outfits.size(1)
    
        self.H_item_user = gen_sparse_directed_H_item(self.users_trajs_dict, num_items)    # [L, L]
        # drop edge on csr_matrix H_pu
        self.H_item_user = csr_matrix_drop_edge(self.H_item_user, args.keep_rate_item)
        self.Deg_H_item_user = get_hyper_deg(self.H_item_user)    # [L, L]
        self.HG_item_user = self.Deg_H_item_user * self.H_item_user    # [L, L]
        self.HG_item_user = transform_csr_matrix_to_tensor(self.HG_item_user).to(device)

        # generate targeted item hypergraph
        self.H_item_outfit = self.H_item_user.T    # [L, L]
        self.Deg_H_item_outfit = get_hyper_deg(self.H_item_outfit)    # [L, L]
        self.HG_item_outfit = self.Deg_H_item_outfit * self.H_item_outfit    # [L, L]
        self.HG_item_outfit = transform_csr_matrix_to_tensor(self.HG_item_outfit).to(device)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


class itemPartialDataset(Dataset):
    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class itemOutfitDataset(Dataset):
    def __init__(self, data_filename, label_filename, outfits_filename, num_items, padding_idx, args, device):

       
        # self.outfits_dict = self.data[0]  # itemID starts from 0
        # self.labels_dict = self.data[1]
        self.outfits_dict = load_dict_from_pkl(data_filename)
        self.labels_dict = load_dict_from_pkl(label_filename)
        self.items_dict = load_dict_from_pkl(outfit_filename)
        self.users_trajs_dict = self.outfits_dict

        # definition
        # self.num_users = num_users
        self.num_items = num_items
        self.num_outfits = len(self.outfits_dict)
        self.padding_idx = padding_idx
        self.attr = args.attr
        self.keep_rate = args.keep_rate
        self.device = device

        # calculate generate adjacency matrix
        self.item_adj = gen_item_adj(num_items, self.items_dict, self.attr)   # csr_matrix
        self.item_graph_matrix = normalized_adj(adj=self.item_adj, is_symmetric=False)
        self.item_graph = transform_csr_matrix_to_tensor(self.item_graph_matrix).to(device)

        self.H_item_user = gen_sparse_directed_H_item(self.users_trajs_dict, num_items)    # [L, L]
        self.H_item_user = csr_matrix_drop_edge(self.H_item_user, args.keep_rate_item)
        self.Deg_H_item_user = get_hyper_deg(self.H_item_user)    # [L, L]
        self.HG_item_user = self.Deg_H_item_user * self.H_item_user    # [L, L]
        self.HG_item_user = transform_csr_matrix_to_tensor(self.HG_item_user).to(device)

        # generate targeted item hypergraph
        self.H_item_outfit = self.H_item_user.T    # [L, L]
        self.Deg_H_item_outfit = get_hyper_deg(self.H_item_outfit)    # [L, L]
        self.HG_item_outfit = self.Deg_H_item_outfit * self.H_item_outfit    # [L, L]
        self.HG_item_outfit = transform_csr_matrix_to_tensor(self.HG_item_outfit).to(device)

        # item-outfit hypergraph
        self.H_item_outfit = gen_sparse_H_items_outfit(self.outfits_dict, num_items, self.num_outfits)
        self.HG_col = gen_HG_from_sparse_H(self.H_item_outfit)
        self.HG_col = transform_csr_matrix_to_tensor(self.HG_col).to(device)

      
        self.H_pu = self.H_item_outfit

        # generate outfit-item incidence matrix, its degree and hypergraph
        self.H_up = self.H_pu.T  # [U, L]
        self.Deg_H_up = get_hyper_deg(self.H_up)  # [U, U]
        self.HG_up = self.Deg_H_up * self.H_up  # [U, L]
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

    def __len__(self):
        # return self.num_users
        return self.num_outfits

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = len(user_seq)
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = user_seq[::-1]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value=3835):
 
    # get each item in the batch
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])

    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    # stack list obj to a torch.tensor
    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    collate_sample = {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
    }

    return collate_sample

