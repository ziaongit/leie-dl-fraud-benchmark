# Run 5: sequence models on genuine provider histories (Physician, grouped repeat 0, 5 folds)
import glob, shutil, os, subprocess
os.chdir('/kaggle/working')
shutil.copy(glob.glob('/kaggle/input/**/p3_bench_r2.py', recursive=True)[0], 'p3_bench.py')
shutil.copy(glob.glob('/kaggle/input/**/p3_hist.py', recursive=True)[0], 'p3_hist.py')
os.system('WORK=/kaggle/working/hist python -u p3_hist.py 2>&1 | grep -v -i warn')
os.system('zip -qr p3v2_run5.zip hist p3_hist.py')
print('RUN5 DONE', flush=True)
