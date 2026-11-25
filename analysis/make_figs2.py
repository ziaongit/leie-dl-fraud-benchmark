import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
R1=os.environ.get('RUN1','results/run1')+'/'; R2=os.environ.get('RUN2','results/run2')+'/'; FIG=os.environ.get('FIG','figures'); os.makedirs(FIG,exist_ok=True)
# Fig 3: attention faithfulness
fig,ax=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
pairs=[('rho_att_shap','Attention vs\nSHAP'),('rho_att_perm','Attention vs\npermutation'),('rho_att_abl','Attention vs\nablation'),('rho_perm_abl','Permutation vs\nablation'),('att_position_corr','Attention vs\nposition')]
for k,ds in enumerate(['phys','partd']):
    a=pd.read_csv(R2+f'p3_attention_results_{ds}.csv'); n=a[a.perm>=0]
    for i,(c,l) in enumerate(pairs):
        v=n[c].values; col='#e07b39' if i<3 else ('#4a6fa5' if i==3 else '#7f7f7f')
        ax[k].boxplot(v,positions=[i],widths=.5,showfliers=False,medianprops=dict(color='k'))
        ax[k].scatter(i+np.random.default_rng(i).uniform(-.15,.15,len(v)),v,s=12,color=col,alpha=.8,zorder=3)
    ax[k].axhline(0,color='grey',lw=.6); ax[k].set_xticks(range(5)); ax[k].set_xticklabels([p[1] for p in pairs],fontsize=7.5)
    ax[k].set_title({'phys':'Physician','partd':'Part D'}[ds]+' (5 folds x 3 feature orders)',fontsize=10); ax[k].spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Spearman correlation')
plt.tight_layout(); plt.savefig(f'{FIG}/Fig3_attention_faithfulness.png',dpi=300)
# Fig 4: learning curve
R=pd.concat([pd.read_csv(R2+f'results_phys_lc{f}.csv').assign(frac=float(f)) for f in ['0.1','0.25','0.5']])
r1=pd.read_csv(R1+'results_phys.csv'); r1=r1[(r1.design=='grouped')&(r1.rep==0)&(r1.perm==0)].assign(frac=1.0); R=pd.concat([R,r1])
M=['LogReg','LightGBM','XGBoost','MLP','FTT','AttnCNNRNN']; lab=['LogReg','LightGBM','XGBoost','MLP','FT-Transf.','Attn CNN-RNN']; col=['#7f7f7f','#1f6f3f','#3a9a5e','#4a6fa5','#6c8ebf','#e07b39']
fig,ax=plt.subplots(1,2,figsize=(9.5,3.5))
for i,m in enumerate(M):
    g=R[R.model==m].groupby('frac')
    ax[0].plot(g.roc_auc.mean().index,g.roc_auc.mean().values,'o-',color=col[i],label=lab[i])
    ax[1].plot(g.lift_pr.median().index,g.lift_pr.median().values,'o-',color=col[i])
for a,t in zip(ax,['ROC-AUC (mean of 5 folds)','PR-AUC lift (median of 5 folds)']):
    a.set_xscale('log'); a.set_xticks([.1,.25,.5,1]); a.set_xticklabels(['10%','25%','50%','100%']); a.set_xlabel('Share of training providers'); a.set_title(t,fontsize=10); a.spines[['top','right']].set_visible(False)
ax[0].legend(fontsize=7.5,frameon=False)
plt.tight_layout(); plt.savefig(f'{FIG}/Fig4_learning_curve.png',dpi=300)
# Fig S1: reliability
fig,ax=plt.subplots(1,2,figsize=(9.5,4))
M7=['LogReg','LightGBM','XGBoost','MLP','FTT','CNNRNN','AttnCNNRNN']; col7=['#7f7f7f','#1f6f3f','#3a9a5e','#4a6fa5','#6c8ebf','#c0504d','#e07b39']
for k,ds in enumerate(['phys','partd']):
    c=pd.read_csv(R1+f'p3_calibration_{ds}.csv').groupby(['model','bin'])[['w','pred','obs']].sum().reset_index()
    for i,m in enumerate(M7):
        x=c[(c.model==m)&(c.w>2000)]; ax[k].plot(x.pred/x.w,np.maximum(x.obs/x.w,1e-6),'o-',ms=3,color=col7[i],label=m.replace('AttnCNNRNN','Attn CNN-RNN').replace('CNNRNN','CNN-RNN').replace('FTT','FT-Transf.'))
    ax[k].plot([1e-5,1],[1e-5,1],'k--',lw=.8); ax[k].set_xscale('log'); ax[k].set_yscale('log'); ax[k].set_xlim(1e-5,1); ax[k].set_ylim(1e-6,1)
    ax[k].set_xlabel('Mean predicted probability'); ax[k].set_title({'phys':'Physician','partd':'Part D'}[ds]+' (grouped, 15 folds pooled)',fontsize=10); ax[k].spines[['top','right']].set_visible(False)
ax[0].set_ylabel('Observed fraud rate'); ax[1].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig(f'{FIG}/FigS1_reliability.png',dpi=300); print('ok')
