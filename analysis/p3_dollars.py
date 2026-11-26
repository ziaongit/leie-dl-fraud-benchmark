"""Audit value in dollars: Medicare payments (Physician: Tot_Mdcr_Pymt_Amt; Part D: Tot_Drug_Cst) of confirmed-fraud
provider-years among the top-k flagged provider-years (grouped design, 15 folds, original feature order). Negatives are
weighted 4 (25% sample). Reports fraud-linked dollars captured, the share of all fraud-linked dollars in the fold, and
the total dollars of all flagged provider-years.  Usage: python p3_dollars.py <results_dir> <dataset>"""
import sys, os, numpy as np, pandas as pd
RD, DS = sys.argv[1], sys.argv[2]
y_all = pd.read_parquet(f'{RD}/_years_{DS}.parquet').label.values.astype(int)
pay = np.nan_to_num(pd.read_parquet(f'{RD}/_pay_{DS}.parquet').pay.values, nan=0.0)
M = ['LogReg', 'LightGBM', 'XGBoost', 'MLP', 'FTT', 'CNNRNN', 'AttnCNNRNN']; rows = []
for rep in range(3):
    for fold in range(5):
        key = f'{DS}_grouped_{rep}_{fold}'; te = np.load(f'{RD}/scores/{key}_test_idx.npy'); y = y_all[te]; p = pay[te]
        w = np.where(y == 1, 1.0, 4.0); fraud_total = (p * y).sum()
        for m in M:
            s = np.load(f'{RD}/scores/{key}_{m}_0.npy'); o = np.argsort(-s); cw = np.cumsum(w[o])
            for k in (1000, 5000):
                top = o[:np.searchsorted(cw, k) + 1]
                cap = (p[top] * y[top]).sum()
                rows.append(dict(dataset=DS, rep=rep, fold=fold, model=m, k=k, fraud_dollars=cap, fraud_dollars_share=cap / fraud_total,
                                 flagged_dollars=(p[top] * w[top]).sum(), fraud_total=fraud_total))
R = pd.DataFrame(rows); R.to_csv(f'{RD}/p3_dollars_{DS}.csv', index=False)
g = R.groupby(['k', 'model']).agg(fraud_usd=('fraud_dollars', 'mean'), share=('fraud_dollars_share', 'mean'), flagged_usd=('flagged_dollars', 'mean'), total=('fraud_total', 'mean'))
print(g.round(3).to_string())
