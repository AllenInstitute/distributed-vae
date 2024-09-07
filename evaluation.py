import glob
from typing import Any, Mapping

from pyrsistent import pmap
import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.dataloader import load_data, get_loaders
from mmidas.eval_models import summarize_inference
from sklearn.metrics.cluster import adjusted_mutual_info_score
from mmidas.utils.tools import get_paths
from tqdm import trange

from dist.plot import noExt, mapv

def mk_vae(saving_folder, input_dim, C, state_dim, arms, latent_dim):
  vae = cpl_mixVAE(saving_folder=saving_folder, device='cpu')
  vae.init_model(n_categories=C,
                     state_dim=state_dim,
                     input_dim=input_dim,
                     lowD_dim=latent_dim,
                     n_arm=arms)
  vae.variational = False
  return vae

def mutinfo(probs, targets):
  preds = np.argmax(probs, axis=1)
  uniq = np.unique(preds)
  
  prediction = np.zeros(len(preds))
  for i, c in enumerate(uniq):
      prediction[np.where(preds == c)[0]] = i
  
  mi = np.zeros((len(np.unique(np.argmax(targets, axis=-1))), 
                 np.shape(uniq)[0]))
  for c in trange(np.shape(mi)[1]):
      per_c_label = np.zeros(len(targets))
      per_c_label[prediction == c] = 1
      for f in range(np.shape(mi)[0]):
          mi[f, c] = adjusted_mutual_info_score(targets[:, f], per_c_label)
  print(mi.shape)
  return mi

def avg(A):
 return np.mean(np.max(A, axis=-1)).item()

def avg_consensus(A): 
  return {
    'all': _avg_consensus_all(A).item(),
    'pairwise': (lambda x: x.item() if isinstance(x, (np.ndarray, np.float64)) else x)(_avg_consensus(A))
  }

def _avg_consensus(A):
  # TODO
  if A.shape[0] == 1:
    return 1.
  total = 0.
  n = 0
  for i in range(A.shape[0]):
    for j in range(i+1, A.shape[0]):
      total += np.mean(A[i] == A[j])
      n += 1
  assert n == A.shape[0] * (A.shape[0] - 1) / 2
  return total / n

def _avg_consensus_all(A):
  return np.mean([sum(np.abs(np.diff(A[:, i]))) == 0 for i in range(A.shape[1])])

def parse_epoch(s: str):
  try:
    return int(noExt(s).split('_epoch_')[-1])
  except:
    return s

def update_key(dct, k, fn, l):
  return dct.set(l, fn(dct[k]))


def parse_toml(tf: str, sf: str) -> Mapping[str, Any]:
  config = get_paths(toml_file=tf, sub_file=sf)
  _trained = config[sf]['trained_model']
  _saving = config['paths']['main_dir'] / config[sf]['saving_path'] / _trained
  _fs = {
    'data': config['paths']['main_dir'] / config[sf]['data_path'] / config[sf]['anndata_file'],
    'saving': _saving,
    'trained': _trained,
  }
  return update_key(pmap(mapv(str, _fs.items())), 'saving', lambda x: x + '/model/cpl_mixVAE_model_before_**', 'pat')

def lookup(ks, dct):
  return [dct[k] for k in ks]

def main():
  SEED = 546
  TOML = 'pyproject.toml'
  SUB = 'mouse_smartseq'
  B = 5000
  DATA = 'log1p'
  TARGETS = 'c_onehot'
  CFG = pmap({
    'arms': 2,
    'C': 92,
    'state_dim': 2,
    'latent_dim': 10,
  })
  RUN = 0
  
  config = parse_toml(TOML, SUB)
  data, targets = lookup([DATA, TARGETS], load_data(config.data))

  preds = summarize_inference(mk_vae(config.saving, data.shape[1], **CFG), 
                              max(glob.glob(config.pat), key=parse_epoch), 
                              get_loaders(data, batch_size=B, seed=SEED)[-1])

  mis = [avg(mutinfo(preds['c_prob'][a], targets.astype(int))) for a in range(CFG.arms)]
  
  consensus = avg_consensus(preds['pred_label'][0])

  assert False
  res = {
    'pairwise': consensus['pairwise'],
    'all': consensus['all'],
    'mi': mis,
    'avg_mi': np.mean(mis).item(),
    'arms': CFG.arms,
  }
  np.save(f'evaluation/A{CFG.arms}-RUN{RUN}.npy', res)
  print(res)

if __name__ == '__main__':
  main()

"""
Let's just use Arm 0 for now

A2R0 vs A2R1
  [] Consensus value
  [] MI value
  [] Consensus plot
  [] MI plot
A2R0 vs A2R2
A2R1 vs A2R2

A2R0 vs ttypes
A2R1 vs ttypes
A2R2 vs ttypes

A3R0 vs A3R1
A3R0 vs A3R2
A3R1 vs A3R2

A3R0 vs ttypes
A3R1 vs ttypes
A3R2 vs ttypes

A5R0 vs A5R1
A5R0 vs A5R2
A5R1 vs A5R2

A5R0 vs ttypes
A5R1 vs ttypes
A5R2 vs ttypes

A2R0 vs A3R0
A2R0 vs A5R0
A3R0 vs A5R0

"""