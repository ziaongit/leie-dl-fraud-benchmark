# Kaggle notebook cells used for the paper (GPU T4 x2). Upload data/phys, data/partd and benchmark/*.py as a Kaggle dataset.
import glob, shutil, os
# --- Run 1: main benchmark (about 6.5 GPU hours) ---
def run1():
    for f in ['p3_bench.py', 'p3_analyse.py']: shutil.copy(glob.glob('/kaggle/input/**/' + f, recursive=True)[0], '/kaggle/working/' + f)
    os.chdir('/kaggle/working')
    for a in ['phys grouped 3 5', 'phys temporal 3', 'partd grouped 3 5', 'partd temporal 3']: os.system('python -u p3_bench.py ' + a)
    os.system('python p3_analyse.py /kaggle/working > analysis.txt 2>&1; zip -qr p3v2_results.zip results_*.csv p3_*.csv scores analysis.txt')
# --- Run 2: attention faithfulness + learning curve (about 1.5 GPU hours) ---
def run2():
    for f in ['p3_bench.py', 'p3_attention.py']: shutil.copy(glob.glob('/kaggle/input/**/' + f, recursive=True)[0], '/kaggle/working/' + f)
    os.chdir('/kaggle/working')
    for ds in ['phys', 'partd']: os.system('ATT_FOLDS=5 python -u p3_attention.py ' + ds)
    for fr in ['0.1', '0.25', '0.5']: os.system('DATA_FRAC=' + fr + ' MODELS=LogReg,LightGBM,XGBoost,MLP,FTT,AttnCNNRNN PERMS=1 python -u p3_bench.py phys grouped 1 5')
    os.system('zip -qr p3v2_run2.zip p3_attention_*.csv results_*.csv scores')
