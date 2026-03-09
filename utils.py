# coding=utf-8

import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def get_unique_seq(outfits_list):
    """Get unique ITEMs in the sequence"""
    seq_list = []
    for outfit in outfits_list:
        for item in outfit:
            if item in seq_list:
                continue
            else:
                seq_list.append(item)

    return seq_list


def get_unique_seqs_for_outfits(outfits_dict):
    """Get unique seq for each outfit"""
    seqs_dict = {}
    seqs_lens_dict = {}
    for key, value in outfits_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_outfits(outfits_dict, padding_idx, max_seq_len):
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    for key, outfits in outfits_dict.items():
        temp = []
        for outfit in outfits:
            temp.extend(outfit)
        if len(temp) >= max_seq_len:
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    with open(filename, 'wb') as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_outfits(outfits_dict):
    num_outfits = 0
    for value in outfits_dict.values():
        num_outfits += len(value)

    return num_outfits


def get_user_complete_traj(outfits_dict):

    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, outfits in outfits_dict.items():
        traj = []
        for outfit in outfits:
            traj.extend(outfit)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


def gen_item_outfit_adj(num_items, items_dict, attr):
    """Generate adjacency matrix with items_dict"""
    item_adj = np.zeros(shape=(num_items, num_items))

    # traverse
    for item1 in range(num_items):
        lat1, lon1 = items_dict[item1]
        for item2 in range(item1, num_items):
            lat2, lon2 = items_dict[item2]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= attr:
                item_adj[item1, item2] = 1
                item_adj[item2, item1] = 1

    # transform np.ndarray to csr_matrix
    item_adj = sp.csr_matrix(item_adj)

    return item_adj


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (max_seq_len - idx)
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for item in seq:
            if item != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):
    """Euclidean distance"""

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_outfit_seqs_adjs_dict(users_seqs_dict, items_dict, max_seq_len, padding_idx, eta=1, attr=2.5, distance_type="haversine"):
    """Generate outfitgraphical sequential adjacency dictionary"""
    outfit_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        outfit_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1 = items_dict[l1]
                lat2, lon2 = items_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= attr:
                    outfit_influence = np.exp(-eta * (dist ** 2))
                    outfit_adj[i, j] = outfit_influence
                    outfit_adj[j, i] = outfit_influence
        outfit_adjs_dict[key] = outfit_adj

    return outfit_adjs_dict


def create_user_item_adj(users_seqs_dict, num_users, num_items):
    """Create user-ITEM interaction matrix"""
    R = sp.dok_matrix((num_users, num_items), dtype=np.float)
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            itemID = itemID - num_users
            R[userID, itemID] = 1

    return R, R.T


def gen_sparse_interaction_matrix(users_seqs_dict, num_users, num_items):
    """Generate sparse user-ITEM adjacent matrix"""
    R, R_T = create_user_item_adj(users_seqs_dict, num_users, num_items)
    A = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj, is_symmetric=True):
    """Normalize adjacent matrix for GCN"""
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj


def normalized_adj_tensor(adj_tensor):
    """Normalized adjacent tensor"""
    # Compute the degree matrix
    degree_tensor = torch.diag(torch.sum(adj_tensor, dim=1))

    # inverse degree
    inverse_degree_tensor = torch.inverse(degree_tensor)

    # normalized adjacency
    norm_adj = torch.matmul(inverse_degree_tensor, adj_tensor)

    sparse_norm_adj = torch.sparse.FloatTensor(norm_adj)

    return sparse_norm_adj


def gen_local_graph(adj):

    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(outfits_dict, num_items, num_outfits, start_itemID):

    H = np.zeros(shape=(num_items, num_outfits))
    sess_idx = 0
    for key, outfits in outfits_dict.items():
        for outfit in outfits:
            for itemID in outfit:
                new_itemID = itemID - start_itemID
                H[new_itemID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_outfits
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_items_outfit(outfits_dict, num_items, num_outfits):
    H = np.zeros(shape=(num_items, num_outfits))
    for sess_idx, outfit in outfits_dict.items():
        for item in outfit:
            H[item, sess_idx] = 1
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user(outfits_dict, num_items, num_users):
    H = np.zeros(shape=(num_items, num_users))

    for userID, outfits in outfits_dict.items():
        seq = []
        for outfit in outfits:
            seq.extend(outfit)
        for item in seq:
            H[item, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_sparse_directed_H_item(users_trajs_dict, num_items):
   
    H = np.zeros(shape=(num_items, num_items))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_item = traj[src_idx]
                tar_item = traj[tar_idx]
                H[src_item, tar_item] = 1
    H = sp.csr_matrix(H)

    return H


def gen_HG_from_sparse_H(H, conv="sym"):
  
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

    return HG


def get_hyper_deg(incidence_matrix):

    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):

    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor


def get_item_outfit_freq(num_items, num_outfits, outfits_dict):
    item_sess_freq_matrix = np.zeros(shape=(num_items, num_outfits))

    # traverse
    sess_idx = 0
    for userID, outfits in outfits_dict.items():
        for outfit in outfits:
            for itemID in outfit:
                item_sess_freq_matrix[itemID, sess_idx] += 1
            sess_idx += 1

    # transform to csr_matrix
    item_sess_freq_matrix = sp.csr_matrix(item_sess_freq_matrix)

    return item_sess_freq_matrix


def get_all_outfits(outfits_dict):

    all_outfits = []

    for userID, outfits in outfits_dict.items():
        for outfit in outfits:
            all_outfits.append(torch.tensor(outfit))

    return all_outfits


def get_all_users_seqs(users_trajs_dict):
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def sparse_adj_tensor_drop_edge(sp_adj, keep_rate):

    if keep_rate == 1.0:
        return sp_adj

    vals = sp_adj._values()
    idxs = sp_adj._indices()
    edgeNum = vals.size()
    mask = ((torch.rand(edgeNum) + keep_rate).floor()).type(torch.bool)
    newVals = vals[mask] / keep_rate
    newIdxs = idxs[:, mask]

    return torch.sparse.FloatTensor(newIdxs, newVals, sp_adj.shape)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]


    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)


    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


