import glob
from pyrsistent import pmap
from functools import lru_cache

import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.dataloader import load_data, get_loaders
from mmidas.eval_models import summarize_inference
from train import parse_toml_
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from mmidas.utils.tools import get_paths

from plot import noExt, mapV

@lru_cache(maxsize=None)
def mkVAE(saving_folder, input_dim, n_categories, state_dim, n_arm, latent_dim):
  vae = cpl_mixVAE(saving_folder=saving_folder, device='cpu')
  vae.init_model(n_categories=n_categories,
                     state_dim=state_dim,
                     input_dim=input_dim,
                     lowD_dim=latent_dim,
                     n_arm=n_arm)
  return vae

def mkMI(c_prob, targets):
  categories = np.argmax(c_prob, axis=1)
  uniq_categories = np.unique(categories)
  model_order = np.shape(uniq_categories)[0]
  n_cluster = len(np.unique(np.argmax(targets, axis=-1)))
  prediction = np.zeros(len(categories))

  mi_mat = np.zeros((n_cluster, model_order))

  for ic, c in enumerate(uniq_categories):
      tmp_idx = np.where(categories == c)[0]
      prediction[tmp_idx] = ic
  
  for c in range(model_order):
      per_c_label = np.zeros(len(targets))
      per_c_label[prediction == c] = 1
      for f in range(n_cluster):
          mi_mat[f, c] = adjusted_mutual_info_score(targets[:, f], per_c_label)

  print(mi_mat.shape)
  return mi_mat

def avgMI(A):
 return np.mean(np.max(A, axis=-1))

def avg_consensus(A): 
  return {
    'all': _avg_consensus_all(A),
    'pairwise': _avg_consensus(A),
  }

def _avg_consensus(A):
  total = 0.
  for i in range(A.shape[0]):
    for j in range(i+1, A.shape[0]):
      total += np.mean(A[i] == A[j])
  return total / (A.shape[0] * (A.shape[0] - 1) / 2)

def _avg_consensus_all(A):
  return np.mean([sum(np.abs(np.diff(A[:, i]))) == 0 for i in range(A.shape[1])])

def parseEpoch(s):
  return int(noExt(s).split('_epoch_')[-1])

def updtK(dct, k, fn, l):
  return dct.set(l, fn(dct[k]))

def mkCfg(tf, sf):
  config = get_paths(toml_file=tf, sub_file=sf)
  _trained = config[sf]['trained_model']
  _saving = config['paths']['main_dir'] / config[sf]['saving_path'] / _trained
  _cfg = {
    'data': config['paths']['main_dir'] / config[sf]['data_path'] / config[sf]['anndata_file'],
    'saving': _saving,
    'trained': _trained,
  }
  return updtK(pmap(mapV(str, _cfg.items())), 'saving', lambda x: x + '/model/cpl_mixVAE_model_**', 'pat')

def main():
  SEED = 546
  TOML = 'pyproject.toml'
  SUB = 'mouse_smartseq'

  cfg = mkCfg(TOML, SUB)

  data = load_data(datafile=cfg.data)

  _, _, ldr = get_loaders(dataset=data['log1p'], batch_size=5000, seed=SEED)

  arms = 5
  n_categories = 92
  state_dim = 2
  latent_dim = 10

  cplMixVAE = mkVAE(cfg.saving, input_dim=data['log1p'].shape[1],
                    n_categories=n_categories, state_dim=state_dim, n_arm=arms, latent_dim=latent_dim)
  cplMixVAE.variational = False 
  last = sorted(glob.glob(cfg.pat), key=parseEpoch)[-1]
  print('model:', last)
  outcome = summarize_inference(cplMixVAE, last, ldr)

  avgMIs = [avgMI(mkMI(outcome['c_prob'][a], data['c_onehot'].astype(int))) 
              for a in range(arms)]
  
  print(avg_consensus(outcome['pred_label'][0]))
  print(f'avgMIs: {avgMIs}')


if __name__ == '__main__':
  main()