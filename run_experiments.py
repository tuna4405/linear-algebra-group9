"""
Run: python run_experiments.py
MovieLens must already exist in data/ml-100k (see README.md).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parent / '.mplconfig'))
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from svd_recommender import SVDRecommender, rating_matrix, metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DATA = ROOT / 'data'
SEEDS = [42, 43, 44]
KS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 943]
SHAPE = (943, 1682)
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 180, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def matrix_csv(path, matrix):
    rows = [{'user': f'U{u+1}', **{f'I{i+1}': '' if np.isnan(v) else round(float(v), 6)
                                for i, v in enumerate(row)}} for u, row in enumerate(matrix)]
    write_csv(path, rows)


def savefig(name):
    plt.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close()


def load_ratings(path):
    records = np.loadtxt(path, dtype=int)[:, :3]
    records[:, :2] -= 1
    return records


def baseline_predictions(model, records, name):
    users, items = records[:, :2].astype(int).T
    if name == 'Global mean':
        return np.full(len(records), model.global_mean)
    if name == 'User mean':
        return model.user_mean[users]
    return model.item_mean[items]


def inner_split(records, seed):
    """Random 80/20 split of development records; never inspect test values.

    No coverage adjustment: empty training users/items are handled explicitly.
    Membership is saved, so each split can be audited exactly.
    """
    perm = np.random.default_rng(seed).permutation(len(records))
    n_validation = len(records) // 5
    validation_ids, train_ids = perm[:n_validation], perm[n_validation:]
    assert not np.intersect1d(train_ids, validation_ids).size
    np.savez_compressed(OUT / f'validation_split_seed_{seed}.npz',
                        train_row_ids=train_ids, validation_row_ids=validation_ids)
    return records[train_ids], records[validation_ids]


def toy_experiment():
    # Hand-authored preferences: U1-U4 favor I1-I4, U5-U8 favor I5-I8.
    truth = np.array([[5,5,4,4,1,1,2,2], [5,4,5,4,1,2,1,2],
                      [4,5,4,5,2,1,2,1], [4,4,5,5,2,2,1,1],
                      [1,1,2,2,5,5,4,4], [1,2,1,2,5,4,5,4],
                      [2,1,2,1,4,5,4,5], [2,2,1,1,4,4,5,5]], float)
    # Hide one liked and one disliked item per user, retaining both taste groups.
    hidden = np.zeros_like(truth, dtype=bool)
    for u in range(8):
        hidden[u, u % 4] = True
        hidden[u, 4 + (u+1) % 4] = True
    train = truth.copy()
    train[hidden] = np.nan
    model = SVDRecommender().fit(train)
    pred = model.reconstruct(2)
    u, i = np.where(hidden)
    result = metrics(truth[hidden], pred[hidden])
    matrix_csv(DATA / 'toy_ground_truth.csv', truth)
    matrix_csv(DATA / 'toy_observed.csv', train)
    for name, matrix in [('toy_imputed', model.filled), ('toy_centered', model.centered), ('toy_predictions_k2', pred)]:
        matrix_csv(OUT / f'{name}.csv', matrix)
    write_csv(OUT / 'toy_input_output.csv', [{'user':int(a+1), 'item':int(b+1),
              'actual':float(truth[a,b]), 'predicted':float(pred[a,b]),
              'absolute_error':float(abs(pred[a,b]-truth[a,b]))} for a,b in zip(u,i)])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, matrix, title in zip(axes, [truth, train, pred],
                                  ['Hand-made ground truth', 'Observed input (48 ratings)', 'Rank-2 predictions']):
        im = ax.imshow(matrix, vmin=1, vmax=5, cmap='YlGnBu')
        for a in range(8):
            for b in range(8):
                label = '?' if np.isnan(matrix[a,b]) else f'{matrix[a,b]:.1f}'
                ax.text(b, a, label, ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(matrix[a,b]) and matrix[a,b] > 3.4 else 'black')
        ax.set(title=title, xlabel='Item', ylabel='User', xticks=range(8), yticks=range(8))
        ax.set_xticklabels(range(1,9)); ax.set_yticklabels(range(1,9))
    fig.colorbar(im, ax=axes, shrink=.8, label='Rating')
    # constrained_layout already manages margins; bbox_inches works on save.
    savefig('toy_example')
    write_csv(OUT / 'toy_recommendations.csv', [{'user':u+1, 'rank':rank+1, 'item':item+1, 'predicted':score}
              for u in range(8) for rank,(item,score) in enumerate(model.recommend(u, 2, 2))])
    return result


def main():
    OUT.mkdir(exist_ok=True)
    toy_result = toy_experiment()
    base_path, test_path = DATA/'ml-100k/u1.base', DATA/'ml-100k/u1.test'
    if not base_path.exists() or not test_path.exists():
        raise FileNotFoundError('Download/extract official MovieLens 100K into data/ml-100k; see README.md')
    development, test = load_ratings(base_path), load_ratings(test_path)
    dev_keys = development[:, 0]*SHAPE[1] + development[:, 1]
    test_keys = test[:, 0]*SHAPE[1] + test[:, 1]
    assert not np.intersect1d(dev_keys, test_keys).size
    assert len(development) == 80000 and len(test) == 20000
    all_rows, timings, diagnostics = [], [], []
    for seed in SEEDS:
        train_records, validation = inner_split(development, seed)
        train = rating_matrix(train_records, SHAPE)
        for imputation in ['item', 'user']:
            print(f'Validation seed={seed}, imputation={imputation}', flush=True)
            start = time.perf_counter()
            model = SVDRecommender(imputation).fit(train)
            timings.append({'stage':'validation', 'seed':seed, 'imputation':imputation,
                            'fit_seconds':time.perf_counter()-start})
            vu, vi = validation[:, :2].T
            tu, ti = train_records[:, :2].T
            # Build residual predictions incrementally: one SVD serves all k.
            v_residual, t_residual = np.zeros(len(validation)), np.zeros(len(train_records))
            previous = 0
            for k in KS:
                v_residual += np.sum(model.U[vu, previous:k]*model.s[previous:k]*model.Vt[previous:k, vi].T, axis=1)
                t_residual += np.sum(model.U[tu, previous:k]*model.s[previous:k]*model.Vt[previous:k, ti].T, axis=1)
                vp = model.user_mean[vu] + v_residual
                vp = np.where(model.user_count[vu] == 0, model.item_mean[vi], vp)
                vp = np.where(model.item_count[vi] == 0, model.user_mean[vu], vp)
                tp = model.user_mean[tu] + t_residual
                val = metrics(validation[:,2], np.clip(vp,1,5))
                training = metrics(train_records[:,2], np.clip(tp,1,5))
                raw_val = metrics(validation[:,2], vp)
                all_rows.append({'imputation':imputation, 'seed':seed, 'k':k,
                                 'train_rmse':training['rmse'], 'train_mae':training['mae'],
                                 'validation_rmse':val['rmse'], 'validation_mae':val['mae'],
                                 'validation_rmse_unclipped':raw_val['rmse'],
                                 'clipped_fraction':float(np.mean((vp<1)|(vp>5))),
                                 'retained_energy':float(np.sum(model.s[:k]**2)/np.sum(model.s**2)),
                                 'factor_scalars':k*(SHAPE[0]+SHAPE[1])})
                previous = k
            if imputation == 'item':
                diagnostics.append({'seed':seed, 'empty_training_users':int(np.sum(model.user_count==0)),
                                    'empty_training_items':int(np.sum(model.item_count==0)),
                                    'validation_ratings_with_cold_item':int(np.sum(model.item_count[vi]==0))})
    write_csv(OUT/'validation_runs.csv', all_rows)
    aggregate = []
    for imputation in ['item','user']:
        for k in KS:
            rows = [r for r in all_rows if r['imputation']==imputation and r['k']==k]
            aggregate.append({'imputation':imputation, 'k':k,
                              **{f'{field}_{stat}':float(function([r[field] for r in rows]))
                                 for field in ['train_rmse','validation_rmse','validation_mae','retained_energy']
                                 for stat,function in [('mean',np.mean),('std',lambda x:np.std(x,ddof=1))]}})
    write_csv(OUT/'validation_summary.csv', aggregate)
    selected = {mode:min([r for r in aggregate if r['imputation']==mode],
                         key=lambda r:(r['validation_rmse_mean'],r['k']))['k'] for mode in ['item','user']}
    print(f'Validation selection frozen: {selected}. Now evaluate final test.', flush=True)
    final_rows, models = [], {}
    for mode in ['item','user']:
        start = time.perf_counter()
        model = SVDRecommender(mode).fit(rating_matrix(development, SHAPE))
        timings.append({'stage':'final_refit', 'seed':'', 'imputation':mode,
                        'fit_seconds':time.perf_counter()-start})
        models[mode] = model
        pred = model.predict(test[:,0], test[:,1], selected[mode])
        final_rows.append({'model':f'SVD ({mode}-mean imputation)', 'k':selected[mode],
                           **metrics(test[:,2],pred)})
        if mode == 'item':
            for baseline in ['Global mean','User mean','Item mean']:
                final_rows.append({'model':baseline, 'k':'', **metrics(test[:,2],baseline_predictions(model,test,baseline))})
            write_csv(OUT/'test_predictions.csv', [{'user':int(u+1), 'item':int(i+1), 'actual':int(actual),
                      'predicted':float(p), 'absolute_error':float(abs(p-actual)),
                      'cold_item':bool(model.item_count[i]==0)} for (u,i,actual),p in zip(test,pred)])
            np.savez_compressed(OUT/'model_item_selected.npz',
                                P=model.factors(selected[mode])[0], Q=model.factors(selected[mode])[1],
                                user_mean=model.user_mean, item_mean=model.item_mean,
                                user_count=model.user_count, item_count=model.item_count,
                                k=selected[mode], rating_range=[1,5])
    write_csv(OUT/'final_test_metrics.csv',final_rows)
    write_csv(OUT/'fit_timings.csv',timings)
    write_csv(OUT/'split_diagnostics.csv',diagnostics)
    final_model = models['item']
    titles = {}
    for line in (DATA/'ml-100k/u.item').read_text(encoding='latin-1').splitlines():
        fields=line.split('|'); titles[int(fields[0])-1]=fields[1]
    write_csv(OUT/'recommendations_user1.csv', [{'user':1, 'rank':rank+1, 'item':item+1,
              'title':titles[item], 'predicted':score} for rank,(item,score) in
              enumerate(final_model.recommend(0,selected['item'],10))])
    write_csv(OUT/'singular_values.csv', [{'component':j+1,'singular_value':float(s),
              'cumulative_energy':float(np.sum(final_model.s[:j+1]**2)/np.sum(final_model.s**2))}
              for j,s in enumerate(final_model.s)])
    for mode in ['item','user']:
        rows = [r for r in aggregate if r['imputation']==mode]
        plt.errorbar([r['k'] for r in rows],[r['validation_rmse_mean'] for r in rows],
                     yerr=[r['validation_rmse_std'] for r in rows],marker='o',markersize=3,
                     capsize=2,label=f'{mode.capitalize()}-mean imputation')
    plt.xscale('symlog',linthresh=3)
    plt.xlabel('Retained dimension k (0 = user-mean baseline)')
    plt.ylabel('Validation RMSE (stars)'); plt.title('Choose k using validation ratings only')
    plt.legend(); plt.grid(alpha=.2); savefig('validation_rmse_vs_k')
    for mode in ['item','user']:
        rows=[r for r in aggregate if r['imputation']==mode]
        plt.plot([r['k'] for r in rows],[r['validation_mae_mean'] for r in rows],'-o',markersize=3,label=mode.capitalize())
    plt.xscale('symlog',linthresh=3); plt.xlabel('Retained dimension k'); plt.ylabel('Validation MAE (stars)')
    plt.title('Absolute prediction error versus k'); plt.legend(); plt.grid(alpha=.2); savefig('validation_mae_vs_k')
    rows=[r for r in aggregate if r['imputation']=='item']
    plt.plot(KS,[r['train_rmse_mean'] for r in rows],'-o',markersize=3,label='Training observed ratings')
    plt.plot(KS,[r['validation_rmse_mean'] for r in rows],'-o',markersize=3,label='Validation ratings')
    plt.xscale('symlog',linthresh=3); plt.xlabel('Retained dimension k'); plt.ylabel('RMSE (stars)')
    plt.title('Reconstruction fit and held-out prediction are different objectives')
    plt.legend(); plt.grid(alpha=.2); savefig('training_vs_validation')
    fig,ax=plt.subplots(figsize=(9,4.4))
    labels=[r['model'].replace(' imputation','') for r in final_rows]
    x=np.arange(len(labels))
    ax.bar(x-.18,[r['rmse'] for r in final_rows],width=.36,label='RMSE')
    ax.bar(x+.18,[r['mae'] for r in final_rows],width=.36,label='MAE')
    ax.set_xticks(x,labels,rotation=12,ha='right'); ax.set_ylabel('Error (stars)')
    ax.set_title('Final test: 20,000 untouched MovieLens ratings'); ax.legend()
    for j,row in enumerate(final_rows):
        ax.text(j-.18,row['rmse']+.01,f"{row['rmse']:.3f}",ha='center',fontsize=8)
        ax.text(j+.18,row['mae']+.01,f"{row['mae']:.3f}",ha='center',fontsize=8)
    ax.set_ylim(0,1.5); savefig('final_test_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].semilogy(range(1,len(final_model.s)+1),final_model.s)
    axes[0].set(xlabel='Component',ylabel='Singular value',title='Spectrum of the imputed centered matrix')
    axes[1].plot(range(1,len(final_model.s)+1),np.cumsum(final_model.s**2)/np.sum(final_model.s**2))
    axes[1].axvline(selected['item'],color='tab:red',linestyle='--',label=f"Selected k={selected['item']}")
    axes[1].set(xlabel='Retained dimension k',ylabel='Cumulative squared singular values',title='Retained matrix energy'); axes[1].legend()
    fig.tight_layout(); savefig('singular_spectrum')
    meta={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
          'shape':list(SHAPE),'development_ratings':len(development),'test_ratings':len(test),
          'inner_training_ratings':64000,'validation_ratings':16000,'seeds':SEEDS,'k_grid':KS,
          'selected_k':selected,'toy':toy_result,'split_diagnostics':diagnostics,
          'final_test_cold_users':int(np.sum(final_model.user_count[test[:,0]]==0)),
          'final_test_cold_items':int(np.sum(final_model.item_count[test[:,1]]==0)),
          'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base_path,test_path]},
          'test_metrics':final_rows,'fit_timings':timings}
    (OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'selected_k':selected,'toy':toy_result,'test':final_rows},indent=2),flush=True)

if __name__=='__main__':
    main()
