import glob
from pyrsistent import pmap

import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.dataloader import load_data, get_loaders
from mmidas.eval_models import summarize_inference
from sklearn.metrics.cluster import adjusted_mutual_info_score
from mmidas.utils.tools import get_paths
from tqdm import trange

from dist.plot import noExt, mapV

def mkVAE(saving_folder, input_dim, C, state_dim, arms, latent_dim):
  vae = cpl_mixVAE(saving_folder=saving_folder, device='cpu')
  vae.init_model(n_categories=C,
                     state_dim=state_dim,
                     input_dim=input_dim,
                     lowD_dim=latent_dim,
                     n_arm=arms)
  vae.variational = False
  return vae

def mkMI(probs, targets):
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

def avgMI(A):
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

def parseEpoch(s):
  try:
    return int(noExt(s).split('_epoch_')[-1])
  except:
    return s

def updtK(dct, k, fn, l):
  return dct.set(l, fn(dct[k]))

def parseTOML(tf, sf):
  config = get_paths(toml_file=tf, sub_file=sf)
  _trained = config[sf]['trained_model']
  _saving = config['paths']['main_dir'] / config[sf]['saving_path'] / _trained
  _fs = {
    'data': config['paths']['main_dir'] / config[sf]['data_path'] / config[sf]['anndata_file'],
    'saving': _saving,
    'trained': _trained,
  }
  return updtK(pmap(mapV(str, _fs.items())), 'saving', lambda x: x + '/model/cpl_mixVAE_model_**', 'pat')

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
    'arms': 7,
    'C': 92,
    'state_dim': 2,
    'latent_dim': 10,
  })
  
  fs = parseTOML(TOML, SUB)
  data, targets = lookup([DATA, TARGETS], load_data(fs.data))

  preds = summarize_inference(mkVAE(fs.saving, data.shape[1], **CFG), 
                              max(glob.glob(fs.pat), key=parseEpoch), 
                              get_loaders(data, batch_size=B, seed=SEED)[-1])

  avgMIs = [avgMI(mkMI(preds['c_prob'][a], targets.astype(int))) for a in range(CFG.arms)]
  
  consensus = avg_consensus(preds['pred_label'][0])
  res = {
    'pairwise': consensus['pairwise'],
    'all': consensus['all'],
    'mi': avgMIs,
    'avg_mi': np.mean(avgMIs).item(),
    'arms': CFG.arms,
  }
  np.save(f'evaluation/A{CFG.arms}.npy', res)
  print(res)

if __name__ == '__main__':
  main()