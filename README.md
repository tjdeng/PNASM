# PNASM: HOW PREDICTORS AFFECT SEARCH STRATEGIES IN NEURAL ARCHITECTURE SEARCH?

## Abstract
Predictor-based Neural Architecture Search is an important topic since it can efficiently reduce the computational cost of evaluating candidate architectures. Most
existing predictor-based NAS algorithms aim to design different predictors to improve the prediction performance. Unfortunately, even a promising performance
predictor may suffer from the accuracy decline due to long-term and continuous
usage, thus leading to the degraded performance of the search strategy. That naturally gives rise to the following problems: how predictors affect search strategies
and how to appropriately use the predictor? In this paper, we take reinforcement
learning (RL) based search strategy to study theoretically and empirically the impact of predictors on search strategies. We first formulate a predictor-RL-based
NAS algorithm as model-based RL and analyze it with a guarantee of monotonic
improvement at each trail. Then, based on this analysis, we propose a simple procedure of predictor usage, named *mixed batch*, which contains ground-truth data
and prediction data. The proposed procedure can efficiently reduce the impact of
predictor errors on search strategies with maintaining performance growth. Our algorithm, Predictor-based Neural Architecture Search with Mixed batch (PNASM),
outperforms traditional NAS algorithms and prior state-of-the-art predictor-based
NAS algorithms on three NAS-Bench-201 tasks and one NAS-Bench-ASR task.

## Requirements
Environments: ensure that Python3.6, PyTorch 1.1.0, and CUDA 9.0 are installed. Then run:
```shell
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html
pip install -r requirements.txt
```

Data: please download the data of natsbench-tss (expansion of nasbench201) [NATS-tss-v1_0-3ffb9-simple](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) and 
[nasbenchasr](https://github.com/SamsungLabs/nb-asr/archive/refs/tags/v1.1.0.tar.gz). Then unzip them and put them into the directory: `data`.

## Running the models
###nasbench201:  
The file `./exps/time_budget_config.yaml` includes all experimental settings of nasbench201. 
Set two fields `dataset` and `search_time` to run experiments on three different tasks of nasbench201.
Set the field `predictor_mode` to `fixed_k` to run the model (PNASM):
```python
python ./exps/main.py --config_file ./exps/time_budget_config.yaml
```

Set the field `predictor_mode` to `adaptive` to run the model (PNASM-A):
```python
python ./exps/main.py --config_file ./exps/time_budget_config.yaml
```

###nasbenchasr:
The file `./exps/search_steps_config.yaml` includes all experimental settings of nasbenchasr. 
The field `target_steps` means if the number of sampled ground-truth architectures reaches `target_steps`, the experiment ends. 
Set the field `predictor_mode` to `fixed_k` to run the model (PNASM):
```python
python ./exps/main.py --config_file ./exps/search_steps_config.yaml
```

Set the field `predictor_mode` to `adaptive` to run the model (PNASM-A):
```python
python ./exps/main.py --config_file ./exps/search_steps_config.yaml
```

## Acknowledgements
To implement this repo, we refer to [NASLib](https://github.com/automl/NASLib) and [NATS-Bench](https://github.com/D-X-Y/NATS-Bench).

