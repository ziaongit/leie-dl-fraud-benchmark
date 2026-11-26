"""Paper 3 v2 - tuning-budget sensitivity (answers the expected "deep models were under-tuned" objection).
Re-runs the provider-grouped benchmark (Physician data, repeat 0 = the same five folds as the main run) with a 4x larger
random search and longer training:
  * 24 configurations per model and split (main run: 6)
  * neural networks up to 30 epochs with early-stopping patience 5 (main run: 10 epochs, patience 3)
  * wider search spaces; the FT-Transformer search includes the size of the published default configuration
    (token dimension 192, 3 layers, learning rate 1e-4; Gorishniy et al., 2021), always evaluated as the first configuration
Everything else (data, preprocessing, training subset, metrics) is p3_bench.py unchanged.
Usage: MODELS=LightGBM,MLP,AttnCNNRNN REPLIST=0 python p3_tune_big.py      (writes to $WORK, default /kaggle/working/big; REPLIST selects grouped repeats)
The number of epochs actually run per configuration is stored in the epochs_run column."""
import os, sys, glob
sys.argv = [sys.argv[0], 'phys', 'grouped', '1', '5']
os.environ.setdefault('N_TRIALS', '24'); os.environ.setdefault('PERMS', '1')
os.environ.setdefault('WORK', '/kaggle/working/big'); os.makedirs(os.environ['WORK'], exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
cands = [os.path.join(here, 'p3_bench.py')] + glob.glob('/kaggle/input/**/p3_bench.py', recursive=True)
src = open([c for c in cands if os.path.exists(c)][0]).read()

def patch(a, b):
    global src
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)

patch("def train_nn(make, P, tri, vai, lr, epochs=10, bs=2048, seed=0):", "EPLOG = []\ndef train_nn(make, P, tri, vai, lr, epochs=30, bs=2048, seed=0):")
patch("            if bad >= 3: break", "            if bad >= 5: break")
patch("    m.load_state_dict(bstate); return m, best", "    EPLOG.append(ep + 1); m.load_state_dict(bstate); return m, best")
# wider search spaces (same number of configurations for every model)
i = src.index("def configs(name, rng):"); j = src.index("def fit_score(")
src = src[:i] + '''def configs(name, rng):
    for k in range(N_TRIALS):
        if name == 'LogReg': yield dict(C=float(10 ** rng.uniform(-4, 1)))
        elif name == 'LightGBM': yield dict(num_leaves=int(rng.choice([15, 31, 63, 127])), learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.1])), min_child_samples=int(rng.choice([20, 50, 200, 500])), n_estimators=int(rng.choice([200, 400, 800])), reg_lambda=float(rng.choice([0, 1, 5, 20])))
        elif name == 'XGBoost': yield dict(max_depth=int(rng.choice([3, 4, 6, 8, 10])), learning_rate=float(rng.choice([0.01, 0.03, 0.08, 0.15])), min_child_weight=float(rng.choice([1, 5, 20, 50])), n_estimators=int(rng.choice([200, 400, 800])), reg_lambda=float(rng.choice([0, 1, 5, 20])))
        elif name == 'MLP': yield dict(h=int(rng.choice([64, 128, 256, 512])), drop=float(rng.choice([0.0, 0.1, 0.3, 0.5])), lr=float(rng.choice([3e-4, 1e-3, 3e-3])))
        elif name == 'FTT':
            if k == 0: yield dict(dim=192, layers=3, lr=1e-4)          # size of the published default configuration
            else: yield dict(dim=int(rng.choice([32, 64, 128, 192])), layers=int(rng.choice([1, 2, 3])), lr=float(rng.choice([1e-4, 3e-4, 1e-3])))
        else: yield dict(ch=int(rng.choice([16, 32, 64, 128])), hid=int(rng.choice([32, 64, 128])), lr=float(rng.choice([3e-4, 1e-3, 3e-3])))

''' + src[j:]
patch("    for rep in range(REPS):\n        salt =", "    for rep in [int(r) for r in os.environ.get('REPLIST', '0').split(',')]:\n        salt =")
patch("t0 = time.time(); best = (-1, None, None)", "t0 = time.time(); best = (-1, None, None); EPLOG.clear()")
patch("            r.update(metrics(y[te], best[1], w_all[te]))", "            r.update(metrics(y[te], best[1], w_all[te])); r['epochs_run'] = json.dumps(EPLOG); r['budget'] = 'big'")
exec(compile(src, 'p3_bench_big', 'exec'))
