"""Builds wan/ecm_wan_kaggle.ipynb: a self-contained Kaggle notebook (code embedded via %%writefile)."""
import json, os, sys
CAL = "--calibrate" in sys.argv

HERE = os.path.dirname(os.path.abspath(__file__))


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("# ECM frame sharing on Wan 2.1 T2V 1.3B\n"
       "Re-noised averaged frame sharing vs fewer sampling steps, measured against the full-compute output "
       "with identical noise. Needs: GPU T4 x2, Internet on. Outputs land in /kaggle/working/ecm_wan."),
    code("!pip -q install diffusers==0.42.1 transformers accelerate ftfy lpips imageio-ffmpeg scikit-image 2>&1 | tail -2\n"
         "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code("%%writefile ecm_wan.py\n" + open(os.path.join(HERE, "ecm_wan.py")).read()),
    code("%%writefile kaggle_run.py\n" + open(os.path.join(HERE, "kaggle_run.py")).read()),
    code("# Settings. CALIBRATE=True runs one prompt with a few methods (timing + threshold check).\n"
         f"CALIBRATE = {CAL}\n"
         "HOURS = 7.0   # per-GPU budget for starting new prompts (Kaggle limit is 12h per run)\n"
         "import torch; NGPU = max(1, torch.cuda.device_count()); print('GPUs:', NGPU)"),
    code("!python kaggle_run.py encode 2>&1 | tail -3"),
    code("import subprocess\n"
         "extra = ['--max_prompts', '1', '--only', 'full_30,unipc_15,share_30_f0.5,share_30_f0.75'] if CALIBRATE else []\n"
         "procs = [subprocess.Popen(['python', 'kaggle_run.py', 'gen', '--gpu', str(g), '--shard', f'{g}/{NGPU}', '--hours', str(HOURS)] + extra,\n"
         "                          stdout=open(f'/kaggle/working/gen_{g}.log', 'w'), stderr=subprocess.STDOUT) for g in range(NGPU)]\n"
         "for p in procs: p.wait()\n"
         "for g in range(NGPU): print(open(f'/kaggle/working/gen_{g}.log').read()[-6000:])"),
    code("!python kaggle_run.py report 2>&1 | grep -v Warning\n"
         "!python - <<'EOF'\nimport json,glob\nfor f in glob.glob('/kaggle/working/ecm_wan/meta_*.json'):\n"
         "    for k,v in json.load(open(f)).items():\n"
         "        if 'act_q' in v: print(k, 'static', round(v['static_frac'],3), 'pan', round(v['pan'],2), 'act_q', [round(q,3) for q in v['act_q']])\nEOF"),
    code("!python kaggle_run.py mp4 && cd /kaggle/working && zip -qr ecm_wan_results.zip ecm_wan/mp4 ecm_wan/*.json ecm_wan/results.md gen_*.log && ls -la ecm_wan_results.zip\n"
         "!rm -rf /kaggle/working/ecm_wan/vids   # raw frames are large; mp4s + metrics are kept"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(os.path.join(HERE, "ecm_wan_kaggle_cal.ipynb" if CAL else "ecm_wan_kaggle.ipynb"), "w"), indent=1)
print("wrote", os.path.join(HERE, "ecm_wan_kaggle.ipynb"))
