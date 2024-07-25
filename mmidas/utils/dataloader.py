import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
import anndata
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from mmidas.utils.tools import get_paths

def string_to_dtype(s):
  if s == 'fp16':
    return torch.float16
  elif s == 'bf16':
    return torch.bfloat16
  elif s == 'fp32':
    return torch.float32
  else:
    raise ValueError(f"Unknown dtype: {s}")

def make_tensor(x, dtype):
    return torch.tensor(x, dtype=string_to_dtype(dtype))

def count_cpu_cores():
    return torch.get_num_threads()

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

def make_data_dir(config):
    return config['paths']['main_dir'] / config['paths']['data_path']

def _make_data_file(toml_file, sub_file):
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_dir = make_data_dir(config)
    return data_dir / config[sub_file]['anndata_file']

def make_data_file(dname):
    if is_smartseq(dname):
        return _make_data_file('pyproject.toml', 'mouse_smartseq')
    else:
        raise ValueError(f'Unknown dataset: {dname}')

def is_smartseq(dname):
    return dname == 'smartseq'

def make_data(dname):
    return load_data(make_data_file(dname))

def din(data):
    # TODO: add predicate "is mmidas data"
    return data['log1p'].shape[1]

def c_onehot_data(data):
    return data['c_onehot']

def c_p_data(data):
    return data['c_p']

def make_dist_sampler(data, rank, world_size, shuffle=True):
    return DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

def make_loaders(data, dname, rank, world_size, use_dist_sampler=False, **config):
    if is_smartseq(dname):
        return _get_loaders(data['log1p'], rank, world_size, use_dist_sampler=use_dist_sampler, **config)
    else:
        raise ValueError(f'Unknown dataset: {dname}')
    
def compute_workers():
    return count_cpu_cores() // 2

def use_persistent_workers(num_workers):
    return num_workers > 0

def make_train_data(data):
    ...

def make_test_data(data):
    ...

def get_sampler(loader):
    return loader.sampler

def is_dist_sampler(sampler):
    return isinstance(sampler, DistributedSampler)
    
def _get_loaders(dataset, rank, world_size, label=[], seed=None, batch_size=128, train_size=0.9, use_dist_sampler=False):
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
    assert torch.allclose(train_set_torch, make_tensor(train_set, 'fp32'))
    train_ind_torch = torch.FloatTensor(train_ind)
    train_data = TensorDataset(train_set_torch, train_ind_torch)

    test_set_torch = torch.FloatTensor(test_set)
    test_ind_torch = torch.FloatTensor(test_ind)
    test_data = TensorDataset(test_set_torch, test_ind_torch)
    # train_loader = DataLoader(train_data, batch_size=batch_size, 
    #                           shuffle=True, drop_last=True, 
    #                           pin_memory=True, num_workers=8,
    #                           persistent_workers=True, prefetch_factor=4)

    if use_dist_sampler:
        train_sampler = make_dist_sampler(train_data, rank, world_size, shuffle=True)
        test_sampler = make_dist_sampler(test_data, rank, world_size, shuffle=True)
    else:
        train_sampler = None
        test_sampler = None

    num_workers = compute_workers()
    cuda_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': use_persistent_workers(num_workers),
    }

    # train_loader = DataLoader(train_data, batch_size=batch_size, 
    #                     shuffle=True, drop_last=True, 
    #                     pin_memory=True)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              sampler=train_sampler, drop_last=True, 
                              **cuda_kwargs)

    # test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler,
                             drop_last=False, **cuda_kwargs)
    return train_loader, test_loader


def get_loaders(dataset, label=[], seed=None, batch_size=128, train_size=0.9):
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
    # train_loader = DataLoader(train_data, batch_size=batch_size, 
    #                           shuffle=True, drop_last=True, 
    #                           pin_memory=True, num_workers=8,
    #                           persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                        shuffle=True, drop_last=True, 
                        pin_memory=True)

    test_set_torch = torch.FloatTensor(test_set)
    test_ind_torch = torch.FloatTensor(test_ind)
    test_data = TensorDataset(test_set_torch, test_ind_torch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=8, persistent_workers=True, prefetch_factor=4)

    data_set_troch = torch.FloatTensor(dataset)
    all_ind_torch = torch.FloatTensor(range(dataset.shape[0]))
    all_data = TensorDataset(data_set_troch, all_ind_torch)
    alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, test_loader, alldata_loader