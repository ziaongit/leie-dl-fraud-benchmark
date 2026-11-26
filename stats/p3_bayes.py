"""Bayesian correlated t-test (Benavoli et al., 2017) for LightGBM vs each deep model over the 15 provider-grouped folds.
Posterior of the mean difference: Student t, df = n-1, location = mean(d), scale = sqrt((1/n + rho/(1-rho)) * var(d)), rho = 1/K = 0.2.
Region of practical equivalence (ROPE): +/- 10% of the mean LightGBM value (the TOST margin of the main analysis); 0.01 for ROC-AUC."""
import pandas as pd, numpy as np
from scipy import stats as st
R=pd.concat([pd.read_csv('/home/claude/check/csv/run1/results_'+d+'.csv') for d in ['phys','partd']]); R=R[(R.perm==0)&(R.design=='grouped')]
rows=[]
for ds in ['phys','partd']:
    for met in ['pr_auc','prec_at_1000','roc_auc']:
        P=R[R.dataset==ds].pivot_table(index=['rep','fold'],columns='model',values=met)
        for m in ['MLP','FTT','CNNRNN','AttnCNNRNN','XGBoost','LogReg']:
            d=(P.LightGBM-P[m]).values; n=len(d); rho=0.2
            sc=np.sqrt((1/n+rho/(1-rho))*d.var(ddof=1)); T=st.t(n-1,loc=d.mean(),scale=sc)
            rope=0.01 if met=='roc_auc' else 0.1*P.LightGBM.mean()
            rows.append(dict(dataset=ds,metric=met,model=m,mean_diff=d.mean(),P_lgbm_better=1-T.cdf(rope),P_rope=T.cdf(rope)-T.cdf(-rope),P_model_better=T.cdf(-rope),P_diff_pos=1-T.cdf(0)))
B=pd.DataFrame(rows); B.to_csv('p3_bayes.csv',index=False)
pd.set_option('display.width',200); print(B.round(3).to_string())
