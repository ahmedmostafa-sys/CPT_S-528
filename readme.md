
DSCT: Diffusion-Enhanced Self-Calibrated Tuning — Reproducible Pipeline_cpt_528
=======================================================================


1. Installation
---------------
Create environment:
    conda create -n dsct python=3.10 -y
    conda activate dsct

Install dependencies:
    pip install -r requirements.txt

Login to HuggingFace:
    huggingface-cli login

2. Prepare Dataset
------------------
Run dataset preparation:
    python prepare_dsct_data.py

This downloads CIFAR-10 and TinyImageNet, producing:
data/
 ├── ID/
 └── OOD_flat/

3. Running in VS Code
---------------------
Open project folder → choose DSCT environment → run:
    python dsct_experiments.py --mode train

4. Training
-----------
Default training:
    python dsct_experiments.py --mode train

Override dataset paths:
    python dsct_experiments.py --mode train --id_root data/ID --ood_root data/OOD_flat

Override class names:
    python dsct_experiments.py --mode train --class_names cat dog car airplane

5. Evaluation
-------------
Run evaluation:
    python dsct_experiments.py --mode eval

Results saved to:
    runs/default/eval_log.csv

6. Ablation Studies
-------------------
Run full ablation grid:
    python dsct_experiments.py --mode ablate --out_dir runs/ablation


// If You Prof. Maung needs to run using yaml file 
7. YAML Configuration
---------------------
Example config.yaml:

device: cuda
epochs: 30
class_names: ["cat", "dog"]
use_dsg: true
use_bpi: true
use_ruc: true

Run:
    python dsct_experiments.py --mode train --config config.yaml

8. Folder Structure
-------------------
.
├── dsct_experiments.py
├── prepare_dsct_data.py
├── requirements.txt
├── config.yaml (optional)
├── data/
│    ├── ID/
│    └── OOD_flat/
└── runs/
     ├── default/
     └── ablation/

9. Example Workflow
-------------------
pip install -r requirements.txt
python prepare_dsct_data.py
python dsct_experiments.py --mode train
python dsct_experiments.py --mode eval

