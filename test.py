import numpy as np


def compute_consensus_iter(labels):
    # confusion matrix code
    c_agreement = []
    for a in range(A):
        pred_a = labels[a]
        for b in range(a + 1, A):
            pred_b = labels[b]
            armA_vs_armB = np.zeros((C, C))

            for samp in range(pred_a.shape[0]):
                armA_vs_armB[
                    pred_a[samp].astype(int), pred_b[samp].astype(int)
                ] += 1

            num_samp_arm = []
            for ij in range(C):
                sum_row = armA_vs_armB[ij, :].sum()
                sum_column = armA_vs_armB[:, ij].sum()
                num_samp_arm.append(max(sum_row, sum_column))

            armA_vs_armB = np.divide(
                armA_vs_armB,
                np.array(num_samp_arm),
                out=np.zeros_like(armA_vs_armB),
                where=np.array(num_samp_arm) != 0,
            )
