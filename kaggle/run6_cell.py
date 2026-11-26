# Run 6 (Kaggle notebook cell, two T4 GPUs): repeats 1-2 of the tuning-budget check and provider histories on 15 folds.
# CPU threads are split between the two processes (OMP/MKL 2 threads, LightGBM/XGBoost n_jobs=2); running both with 4 threads
# each on the 4-core Kaggle machine slowed LightGBM about 25-fold.
import glob, shutil, os, subprocess, time; os.chdir('/kaggle/working'); F = lambda f: sorted(glob.glob(f'/kaggle/input/**/{f}', recursive=True))[-1]
shutil.copy(F('p3_bench_r2.py'), 'p3_bench.py'); shutil.copy(F('p3_tune_big.py'), 'p3_tune_big.py'); shutil.copy(F('p3_hist.py'), 'p3_hist.py'); os.system("sed -i 's/n_jobs=4/n_jobs=2/g' p3_bench.py p3_hist.py")
E = 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
p0 = subprocess.Popen(E + 'CUDA_VISIBLE_DEVICES=0 WORK=/kaggle/working/big0 MODELS=MLP,AttnCNNRNN REPLIST=1,2 python -u p3_tune_big.py > big0.log 2>&1', shell=True)
p1 = subprocess.Popen('(' + E + 'CUDA_VISIBLE_DEVICES=1 WORK=/kaggle/working/hist HDS=phys REPLIST=1,2 python -u p3_hist.py; ' + E + 'CUDA_VISIBLE_DEVICES=1 WORK=/kaggle/working/hist HDS=partd REPLIST=0,1,2 python -u p3_hist.py; ' + E + 'CUDA_VISIBLE_DEVICES=1 WORK=/kaggle/working/bigL MODELS=LightGBM REPLIST=1,2 python -u p3_tune_big.py) > hist.log 2>&1', shell=True)
while p0.poll() is None or p1.poll() is None: time.sleep(300); os.system("grep -h -E '^grouped|^hist ' big0.log hist.log | tail -n 3")
os.system("grep -h -E '^grouped|^hist |Traceback|Error' big0.log hist.log"); os.system('zip -qr p3v2_run6.zip big0 bigL hist/results_hist_phys.csv hist/results_hist_partd.csv big0.log hist.log'); print('RUN6 DONE')
