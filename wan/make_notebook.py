"""Builds wan/ecm_wan_kaggle.ipynb: a self-contained Kaggle notebook (code embedded via %%writefile)."""
import json, os, sys
CAL = "--calibrate" in sys.argv
V3 = "--v3" in sys.argv

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
         + ("HOURS = 4.0   # per-GPU budget for starting new prompts\n" if V3 else
            "HOURS = 7.5   # per-GPU budget for starting new prompts (Kaggle limit is 12h per run)\n")
         + ("V3ARGS = ['--set', 'v3', '--max_prompts', '4']\n" if V3 else "V3ARGS = []\n")
         + "import torch; NGPU = max(1, torch.cuda.device_count()); print('GPUs:', NGPU)"),
    code("!python kaggle_run.py encode 2>&1 | tail -3"),
    code("import subprocess\n"
         "extra = ['--max_prompts', '1', '--only', 'full_30,unipc_15,share_30_f0.75_draft3'] if CALIBRATE else []\n"
         "procs = [subprocess.Popen(['python', 'kaggle_run.py', 'gen', '--gpu', str(g), '--shard', f'{g}/{NGPU}', '--hours', str(HOURS)] + V3ARGS + extra,\n"
         "                          stdout=open(f'/kaggle/working/gen_{g}.log', 'w'), stderr=subprocess.STDOUT) for g in range(NGPU)]\n"
         "for p in procs: p.wait()\n"
         "for g in range(NGPU): print(open(f'/kaggle/working/gen_{g}.log').read()[-6000:])"),
    code("!python kaggle_run.py diag 2>&1 | grep -v Warning | tail -40" if V3 else "print('no diag')"),
    code("!python kaggle_run.py report 2>&1 | grep -v Warning\n"
         "import json, glob\n"
         "for f in glob.glob('/kaggle/working/ecm_wan/meta_*.json'):\n"
         "    for k, v in json.load(open(f)).items():\n"
         "        print(k, round(v['seconds'], 1), 's', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a in ('static_frac', 'pan', 'kept_frac', 'act_q')})"),
    code("!python kaggle_run.py mp4 && cd /kaggle/working && zip -qr ecm_wan_results.zip ecm_wan/mp4 ecm_wan/*.json ecm_wan/*.md gen_*.log && ls -la ecm_wan_results.zip\n"
         "!rm -rf /kaggle/working/ecm_wan/vids   # raw frames are large; mp4s + metrics are kept"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(os.path.join(HERE, "ecm_wan_kaggle_cal.ipynb" if CAL else ("ecm_wan_kaggle_v3.ipynb" if V3 else "ecm_wan_kaggle.ipynb")), "w"), indent=1)
print("wrote", os.path.join(HERE, "ecm_wan_kaggle.ipynb"))
