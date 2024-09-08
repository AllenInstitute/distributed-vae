import numpy as np
import pickle
import matplotlib.pyplot as plt
from .cpl_mixvae import cpl_mixVAE


def summarize_inference(cpl: cpl_mixVAE, files, data, saving_folder=''):
    
    """
        Inference summary for the cpl model

    input args
        cpl: the cpl_mixVAE class object.
        files: the list of model files to be evaluated.
        data: the input data loader.
        saving_folder: the path to save the output dictionary.

    return
        data_dic: the output dictionary containing the summary of the inference.
    """

    A = cpl.n_arm
    C = cpl.n_categories

    recon_loss = []
    label_pred = []
    test_dist_c = []
    test_dist_qc = []
    n_pruned = []
    consensus_min = []
    consensus_mean = []
    test_loss = [[] for _ in range(A)]
    prune_indx = []
    consensus = []
    a_vs_b = []
    sample_id = []
    data_rec = []

    files = [files] if not isinstance(files, list) else files

    for i, file in enumerate(files):
        print(f'Model {file[file.rfind('/'):]}')
        cpl.load_model(file)
        evals = cpl.eval_model(data)

        x_low = evals['x_low']
        predicted_label = evals['predicted_label']
        test_dist_c.append(evals['total_dist_z'])
        test_dist_qc.append(evals['total_dist_qz'])
        recon_loss.append(evals['total_loss_rec'])
        c_prob = evals['z_prob']
        prune_indx.append(evals['prune_indx'])
        sample_id.append(evals['data_indx'])
        label_pred.append(predicted_label)

        for a in range(A):
            test_loss[a].append(evals['total_loss_rec'][a])

        if cpl.ref_prior:
            A += 1

        for a in range(A):
            pred_a = predicted_label[a, :]
            for b in range(a + 1, A):
                pred_b = predicted_label[b, :]
                _a_vs_b = np.zeros((C, C))

                for samp in range(pred_a.shape[0]):
                    _a_vs_b[pred_a[samp].astype(int) - 1, pred_b[samp].astype(int) - 1] += 1

                num_samp_arm = []
                for c in range(C):
                    num_samp_arm.append(max(_a_vs_b[c, :].sum(), _a_vs_b[:, c].sum()))

                nprune_indx = np.where(np.isin(range(C), prune_indx[i]) == False)[0]
                _consensus = np.divide(_a_vs_b, np.array(num_samp_arm), out=np.zeros_like(_a_vs_b),
                                         where=np.array(num_samp_arm) != 0)[:, nprune_indx][nprune_indx]
                _a_vs_b = _a_vs_b[:, nprune_indx][nprune_indx]

                consensus.append(_consensus)
                consensus_min.append(np.min(np.diag(_consensus)))
                consensus_mean.append(1. - (sum(np.abs(predicted_label[0, :] - predicted_label[1, :]) > 0.) / predicted_label.shape[1]))
                a_vs_b.append(_a_vs_b)
                

        # TODO: check
        if A == 1:
            nprune_indx = np.where(np.isin(range(C), prune_indx[i]) == False)[0]
        n_pruned.append(list(range(C)))
        plt.close()

    summary = {
        'recon_loss': test_loss,
        'dc': test_dist_c,
        'd_qc': test_dist_qc,
        'con_min': consensus_min,
        'con_mean': consensus_mean,
        'num_pruned': n_pruned,
        'pred_label': label_pred,
        'consensus': consensus,
        'armA_vs_armB': a_vs_b,
        'prune_indx': prune_indx,
        'nprune_indx': nprune_indx,
        'state_mu': evals['state_mu'],
        'state_sample': evals['state_sample'],
        'state_var': evals['state_var'],
        'sample_id': sample_id,
        'c_prob': c_prob,
        'lowD_x': x_low,
        'x_rec': data_rec
    }

    if saving_folder:
        f_name = saving_folder + '/summary_performance_K_' + str(C) + '_narm_' + str(A) + '.p'
        f = open(f_name, "wb")
        pickle.dump(summary, f)
        f.close()

    return summary