import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot(df_groups, alphas, n_samples_list, logy=False, shap=False):
    fig, ax = plt.subplots(ncols=len(n_samples_list), figsize=(12, 6), dpi=80)

    for j, (n_samples, (df_group_imp, df_group_performance)) in enumerate(zip(n_samples_list, df_groups)):
        for i in range(20):
            c = 'C0' if i < 10 else 'C1'
            ax[j].errorbar(df_group_imp.index, df_group_imp[f"column_{i+1}_mean"],
                           df_group_imp[f"column_{i+1}_std"],
                           c=c)

        if logy:
            ax[j].set_yscale('log')
        else:
            if not shap:
                ax[j].set_ylim(0, 0.4)

        ax[j].set_xscale('log')
        ax[j].set_xticks(alphas, alphas, rotation=90)
        ax[j].set_title(f'{n_samples=}')
        ax[j].set_xlabel('alpha')
        ax[j].set_ylabel('Feature importances')
        ax2 = ax[j].twinx()
        ax2.errorbar(df_group_performance.index, df_group_performance.ROCAUC_mean,
                     df_group_performance.ROCAUC_std, c='k')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Performance (ROC AUC)')

    plt.tight_layout()

def undersample(X, y, alpha, random_state=42):
    size = int((y==0).sum()*alpha)
    try:
        return (
            np.vstack([X[y==0], pd.DataFrame(X[y==1]).sample(n=size, replace=False, random_state=random_state)]),
            np.hstack([y[y==0], np.array(size*[1])])
        )
    except:
        return X, y