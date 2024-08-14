import glob
from pyrsistent import pmap

import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.dataloader import load_data, get_loaders
from mmidas.eval_models import summarize_inference
from train import parse_toml_
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from mmidas.utils.tools import get_paths


def mk_vae(saving_folder, input_dim, n_categories, state_dim, n_arm, latent_dim):
  vae = cpl_mixVAE(saving_folder=saving_folder, device='cpu')
  vae.init_model(n_categories=n_categories,
                     state_dim=state_dim,
                     input_dim=input_dim,
                     lowD_dim=latent_dim,
                     n_arm=n_arm)
  return vae


def mk_mi_mat(c_prob, targets):
  mi_ind = []

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

def avg_mi(A):
 return np.mean(np.max(A, axis=-1))

def avg_consensus(A): 
  return {
    'all': _avg_consensus_all(A),
    'pairwise': _avg_consensus(A),
  }

def _avg_consensus(A):
  # return np.mean(np.max(A, axis=-1))
  total = 0.
  for i in range(A.shape[0]):
    for j in range(i+1, A.shape[0]):
      # agree += sum(A[i] == A[j])/ len(A[i])
      total += np.mean(A[i] == A[j])
  return total / (A.shape[0] * (A.shape[0] - 1) / 2)

def _avg_consensus_all(A):
  return np.mean([sum(np.abs(np.diff(A[:, i]))) == 0 for i in range(A.shape[1])])

def main():
  toml_file = 'pyproject.toml'
  sub_file = 'mouse_smartseq'
  config = get_paths(toml_file=toml_file, sub_file=sub_file)
  data_path = config['paths']['main_dir'] / config[sub_file]['data_path']
  data_file = data_path / config[sub_file]['anndata_file']
  data = load_data(datafile=data_file)

  trainloader, testloader, all_dataloader = get_loaders(dataset=data['log1p'],
                                                         batch_size=5000, seed=0)
  saving_folder = config['paths']['main_dir'] / config[sub_file]['saving_path']
  trained_model_folder = config[sub_file]['trained_model']
  saving_folder = str(saving_folder / trained_model_folder)

  arms = 10
  n_categories = 92
  state_dim = 2
  latent_dim = 10
  
  cplMixVAE = mk_vae(saving_folder, input_dim=data['log1p'].shape[1],
                       n_categories=n_categories, state_dim=state_dim, n_arm=arms, latent_dim=latent_dim)
  cplMixVAE.variational = False 
  selected_model = sorted(glob.glob(saving_folder + '/model/cpl_mixVAE_model_**'))[-1]
  print('model:', selected_model)
  outcome = summarize_inference(cplMixVAE, selected_model, all_dataloader)

  avg_mis = [avg_mi(mk_mi_mat(outcome['c_prob'][a], data['c_onehot'].astype(int))) 
              for a in range(arms)]
  
  print(avg_consensus(outcome['pred_label'][0]))
  print(f'avg_mis: {avg_mis}')


if __name__ == '__main__':
  main()