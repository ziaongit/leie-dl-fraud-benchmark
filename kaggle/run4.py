# Run 4: tuning-budget sensitivity (24 configurations, up to 30 epochs); two T4 GPUs in parallel
import glob, shutil, os, subprocess, time
os.chdir('/kaggle/working')
shutil.copy(glob.glob('/kaggle/input/**/p3_bench_r2.py', recursive=True)[0], 'p3_bench.py')
shutil.copy(glob.glob('/kaggle/input/**/p3_tune_big.py', recursive=True)[0], 'p3_tune_big.py')
p0 = subprocess.Popen('CUDA_VISIBLE_DEVICES=0 WORK=/kaggle/working/big0 MODELS=LightGBM,MLP,AttnCNNRNN python -u p3_tune_big.py > big0.log 2>&1', shell=True)
p1 = subprocess.Popen('CUDA_VISIBLE_DEVICES=1 WORK=/kaggle/working/big1 MODELS=FTT python -u p3_tune_big.py > big1.log 2>&1', shell=True)
while p0.poll() is None or p1.poll() is None:
    time.sleep(300); os.system('grep -h "grouped" big0.log big1.log | tail -n 4')
os.system('grep -v -i warn big0.log big1.log | tail -n 40')
os.system('zip -qr p3v2_run4.zip big0/results_phys.csv big1/results_phys.csv big0.log big1.log p3_tune_big.py')
print('RUN4 DONE', flush=True)
