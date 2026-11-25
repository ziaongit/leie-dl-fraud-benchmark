import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
RD=os.environ.get('RUN1','results/run1')+'/'; FIG=os.environ.get('FIG','figures'); os.makedirs(FIG,exist_ok=True)
R=pd.concat([pd.read_csv(RD+f) for f in ['results_phys.csv','results_partd.csv']])
M=['LogReg','LightGBM','XGBoost','MLP','FTT','CNNRNN','AttnCNNRNN']; lab=['LogReg','LightGBM','XGBoost','MLP','FT-Transf.','CNN-RNN','Attn CNN-RNN']
col=['#7f7f7f','#1f6f3f','#3a9a5e','#4a6fa5','#6c8ebf','#c0504d','#e07b39']
R0=R[R.perm==0]
fig,ax=plt.subplots(1,4,figsize=(13,3.6),sharey=False)
for k,(ds,des) in enumerate([('phys','grouped'),('phys','temporal'),('partd','grouped'),('partd','temporal')]):
    s=R0[(R0.dataset==ds)&(R0.design==des)]
    for i,m in enumerate(M):
        v=s[s.model==m].lift_pr.values
        ax[k].bar(i,v.mean(),color=col[i]); ax[k].scatter(np.full(len(v),i)+np.random.default_rng(i).uniform(-.2,.2,len(v)),v,s=6,color='k',alpha=.5,zorder=3)
    ax[k].set_xticks(range(7)); ax[k].set_xticklabels(lab,rotation=55,ha='right',fontsize=8)
    ax[k].set_title({'phys':'Physician','partd':'Part D'}[ds]+', '+des,fontsize=10); ax[k].spines[['top','right']].set_visible(False)
ax[0].set_ylabel('PR-AUC lift over base rate')
plt.tight_layout(); plt.savefig(f'{FIG}/Fig1_benchmark_lift.png',dpi=300)
# Fig 2 order sensitivity
S=R[(R.rep==0)&(R.model.isin(['CNNRNN','AttnCNNRNN']))&(R.design=='grouped')]
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for k,ds in enumerate(['phys','partd']):
    for j,m in enumerate(['CNNRNN','AttnCNNRNN']):
        for f in range(5):
            v=S[(S.dataset==ds)&(S.model==m)&(S.fold==f)].sort_values('perm').lift_pr.values
            x=f+(-.15 if j==0 else .15)
            ax[k].plot([x]*len(v),v,'-',color=['#c0504d','#e07b39'][j],alpha=.6); ax[k].scatter([x]*len(v),v,c=['#c0504d','#e07b39'][j],marker=['o','s','^'][0],s=18,zorder=3)
        ax[k].plot([],[],'o-',color=['#c0504d','#e07b39'][j],label=['CNN-RNN','Attn CNN-RNN'][j])
    ax[k].set_xticks(range(5)); ax[k].set_xticklabels([f'Fold {i+1}' for i in range(5)],fontsize=8)
    ax[k].set_title({'phys':'Physician','partd':'Part D'}[ds]+' (grouped, repeat 1)',fontsize=10); ax[k].spines[['top','right']].set_visible(False)
ax[0].set_ylabel('PR-AUC lift (3 feature orders)'); ax[1].legend(fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig(f'{FIG}/Fig2_order_sensitivity.png',dpi=300)
print('ok')
