import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
import anndata
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def load_data(datafile, n_gene=0, gene_id=[], rmv_type=[], min_num=10, eps=1e-1, tau=1.0):

    adata = anndata.read_h5ad(datafile)
    print('data is loaded!')

    data = dict()
    data['log1p'] = adata.X
    data['gene_id'] = np.array(adata.var.index)
    anno_key = adata.obs.keys()
    for key in anno_key:
        data[key] = adata.obs[key].values
        if key == 'cluster':
            data['cluster_label'] = adata.obs[key].values

    if n_gene == 0:
        n_gene = len(data['gene_id'])

    if len(gene_id) > 0:
        gene_idx = [np.where(data['gene_id'] == gg)[0] for gg in gene_id]
        gene_idx = np.concatenate(gene_idx).astype(int)
        data['gene_id'] = data['gene_id'][gene_idx]
        data['log1p'] = data['log1p'][:, gene_idx].todense()
        n_gene = len(gene_idx)
    else:
        data['gene_id'] = data['gene_id'][:n_gene]
        data['log1p'] = data['log1p'][:, :n_gene].todense()

    for ttype in rmv_type:
        for key in data:
            if key not in ['gene_id']:
                data[key] = np.delete(data[key], ind, axis=0)

    uniq_clusters = np.unique(data['cluster_label'])
    count = np.zeros(len(uniq_clusters))
    for it, tt in enumerate(uniq_clusters):
        count[it] = sum(data['cluster_label'] == tt)

    uniq_clusters = uniq_clusters[count >= min_num]
    data['cluster_id'] = np.zeros(len(data['cluster_label']))
    for ic, cls_ord in enumerate(np.unique(data['cluster_label'])):
        data['cluster_id'][data['cluster_label'] == cls_ord] = int(ic + 1)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['cluster_id'])
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    data['c_onehot'] = onehot_encoder.fit_transform(integer_encoded)
    data['c_p'] = softmax((data['c_onehot'] + eps) / tau, axis=1)
    data['n_type'] = len(np.unique(data['cluster_label']))

    print(' --------- Data Summary --------- ')
    print(f'num cell types: {len(np.unique(data["cluster_label"]))}, num cells: {data["log1p"].shape[0]}, num genes:{len(data["gene_id"])}')
    
    return data


def data_gen(dataset, train_size, seed):

        test_size = dataset.shape[0] - train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(
            dataset, np.arange(dataset.shape[0]), train_size=train_size, test_size=test_size, random_state=seed)

        return train_cpm, test_cpm, train_ind, test_ind


def get_loaders(dataset, label=[], seed=None, batch_size=128, train_size=0.9):

        batch_size = batch_size

        if len(label) > 0:
            train_ind, val_ind, test_ind = [], [], []
            for ll in np.unique(label):
                indx = np.where(label == ll)[0]
                tt_size = int(train_size * sum(label == ll))
                _, _, train_subind, test_subind = data_gen(dataset, tt_size, seed)
                train_ind.append(indx[train_subind])
                test_ind.append(indx[test_subind])

            train_ind = np.concatenate(train_ind)
            test_ind = np.concatenate(test_ind)
            train_set = dataset[train_ind, :]
            test_set = dataset[test_ind, :]
        else:
            tt_size = int(train_size * dataset.shape[0])
            train_set, test_set, train_ind, test_ind = data_gen(dataset, tt_size, seed)

        train_set_torch = torch.FloatTensor(train_set)
        train_ind_torch = torch.FloatTensor(train_ind)
        train_data = TensorDataset(train_set_torch, train_ind_torch)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

        test_set_torch = torch.FloatTensor(test_set)
        test_ind_torch = torch.FloatTensor(test_ind)
        test_data = TensorDataset(test_set_torch, test_ind_torch)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

        data_set_troch = torch.FloatTensor(dataset)
        all_ind_torch = torch.FloatTensor(range(dataset.shape[0]))
        all_data = TensorDataset(data_set_troch, all_ind_torch)
        alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return train_loader, test_loader, alldata_loader