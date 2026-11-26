# Run 7: independent labels from 32 state Medicaid exclusion lists (Physician on GPU 0, Part D on GPU 1), all 7 models
import glob, shutil, os, subprocess, time
os.chdir('/kaggle/working')
shutil.copy(glob.glob('/kaggle/input/**/p3_bench_r2.py', recursive=True)[0], 'p3_bench.py')
shutil.copy(glob.glob('/kaggle/input/**/p3_states.py', recursive=True)[0], 'p3_states.py')
p0 = subprocess.Popen('CUDA_VISIBLE_DEVICES=0 SDS=phys WORK=/kaggle/working/states python -u p3_states.py > states_phys.log 2>&1', shell=True)
p1 = subprocess.Popen('CUDA_VISIBLE_DEVICES=1 SDS=partd WORK=/kaggle/working/states python -u p3_states.py > states_partd.log 2>&1', shell=True)
while p0.poll() is None or p1.poll() is None:
    time.sleep(120); os.system('grep -h -E "^states|positives" states_phys.log states_partd.log | tail -n 4')
os.system('grep -h -E "^states|state-only|positives by|Traceback|Error" states_phys.log states_partd.log')
os.system('python -c "import pandas as pd,glob; d=pd.concat([pd.read_csv(f) for f in glob.glob(\'states/p3_states_*.csv\')]); print(d[[\'dataset\',\'model\',\'scope\',\'positives\',\'roc_auc\',\'lift_pr\',\'prec_at_1000\',\'prec_at_5000\']].round(4).to_string())"')
os.system('zip -qr p3v2_run7.zip states/p3_states_*.csv states_phys.log states_partd.log')
print('RUN7 DONE', flush=True)
