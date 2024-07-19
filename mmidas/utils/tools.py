import numpy as np
import scipy.io as sio
import toml
import requests
from pathlib import Path
from sklearn.preprocessing import normalize

def get_paths(toml_file, sub_file='files', verbose=False):
    """Loads dictionary with path names and any other variables set through xxx.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """

<<<<<<< HEAD
    # package_dir = Path().resolve().parents[1]
    package_dir = Path().resolve()
=======
    package_dir = Path().resolve() #.parents[1]
>>>>>>> origin/main
    config_file = package_dir / toml_file
    print(config_file)

    if not Path(config_file).is_file():
        print(f'Did not find project`s toml file: {config_file}')

    f = open(config_file, "r")
    config = toml.load(f)
    f.close()

    config['paths'].update({'main_dir': package_dir})

    if verbose:
        for key in config.keys():
            print(f'{key}: {config[key]}')

    for key in config:
        if key=='paths':
            for key2 in config['paths']:
                if Path(config['paths'][key2]).exists():
                    config['paths'][key2] = Path(config['paths'][key2])
        if key==sub_file:
            print(f'Getting files directories belong to {sub_file}...')
            for key2 in config[sub_file]:
                if Path(config[sub_file][key2]).exists():
                    config[sub_file][key2] = Path(config[sub_file][key2])

    return config

def normalize_cellxgene(x) -> np.array:
    """ Normalize based on number of input genes

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
    
    return
        normalized gene expression matrix
    """
    return normalize(x, axis=1, norm='l1')


def logcpm(x, scaler=1e6) -> np.array:
    """ Log CPM normalization

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scaler (float, optional): scaling factor for log CPM
    
    return 
        normalized log CPM gene expression matrix
    """
    return np.log1p(normalize_cellxgene(x) * scaler)


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return


def reorder_genes(x, chunksize=1000, eps=1e-1):
    t_gene = x.shape[1]
    print(t_gene)
    g_std, g_bin_std = [], []

    for iter in range(int(t_gene // chunksize) + 1):
        ind0 = iter * chunksize
        ind1 = np.min((t_gene, (iter + 1) * chunksize))
        x_bin = np.where(x[:, ind0:ind1] > eps, 1, 0)
        g_std.append(np.std(x[:, ind0:ind1], axis=0))
        g_bin_std.append(np.std(x_bin, axis=0))

    g_std = np.concatenate(g_std)
    g_bin_std = np.concatenate(g_bin_std)
    g_ind = np.argsort(g_bin_std)
    g_ind = g_ind[np.sort(g_bin_std) > eps]
    print(len(g_ind))
    return g_ind[::-1]


def download_file(url, local_filename, chunk_size=10000):
    """Download a file from a URL and save it locally

    Args:
        url (str): URL of the file to download
        local_filename (str): Local path to save the file
        chunk_size (int, optional): Size of the chunks to download
    """
    # Send a HTTP GET request to the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check if the request was successful
        # Open a local file in binary write mode
        with open(local_filename, 'wb') as file:
            # Stream the content and write it in chunks to the local file
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)



