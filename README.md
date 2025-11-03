# Variational Uncertainty Decomposition for In-Context Learning
> by [I. Shavindra Jayasekera](http://yingzhenli.net/home/en/?page_id=1411)\*, [Jacob Si](https://jacobyhsi.github.io/)\*, [Filippo Valdettaro](https://faisallab.org/members/filippo-valdettaro), [Wenlong Chen](https://chenw20.github.io/wenlongchen.github.io//), [Aldo Faisal](https://faisallab.org/members/aldo-faisal), and [Yingzhen Li](http://yingzhenli.net/home/en/).
> Imperial College London.

<p align="center">
<a href="https://arxiv.org/abs/2509.02327"><img src="https://img.shields.io/badge/arXiv-2505.18495-b31b1b.svg?logo=arxiv&logoColor=red" alt="VUD on arXiv"/></a>
<a href="https://github.com/jacobyhsi/VUD/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

<div align="center">
  <img src="imgs/overview.png" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 1: Uncertainty Decomposition with Auxiliary Data (Above). Decomposition for Two-Moons Dataset (Below).</em>
</div>

<div align="center">
  <img src="imgs/framework.png" alt="Model Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <p><em>Figure 2: Variational Uncertainty Decomposition (VUD) Framework.</em>
</div>

## Installation

The following delineates the installation instructions. Clone this repository and navigate to it in your terminal. Create an environment using a preferred package manager.

Note: can replace `conda` with `micromamba` or `uv`.

```
conda create -n vud python=3.10

conda activate vud

pip install vllm
pip install ipykernel
pip install -U ipywidgets
pip install nbconvert
pip install accelerate
pip install pandas matplotlib datasets scikit-learn flask
pip install gpytorch botorch
```

## Serving the LLM
To run an experiment, first, serve the language model in a terminal.

```
bash run_llm.sh
```

Then in a different terminal, run the desired experiments.

## Experiments

### Toy Datasets

Example Scripts:

```
python run_toy_classification.py --dataset_name [NAME_OF_DATASET]

python run_toy_regression.py --dataset_name [NAME_OF_DATASET]
```

Parameters:

API Parameters
- `model_name`: The name of the model to use for predictions. Options: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B` and `meta-llama/Meta-Llama-3-8B`. `Qwen/Qwen2.5-14B` is the default.
- `model_port`: The port number for the model server. Default is `8000`.
- `model_ip`: The IP address of the model server. Default is `localhost`.
- `model_temperature`: The temperature for the model. Default is `1.0`.
- `is_local_client`: Whether to use a local client for the model. Default is `1` (True). `0` for OpenAI API.

Dataset Parameters
- `dataset_name`: The name of the dataset to use. Options: `logistic_regression`, `moons_1`, `moons_2`, `spirals`, `linear_regression` `gaps`. Default is `logistic_regression` for `toy_classification.py` and `linear_regression` for `toy_regression.py`.
- `D_size`: The size of the dataset D. Default is `15`.

X Parameters
- `x_row_method`: The method to use for generating the x row. Options: `x_range`, `x_features`, `sample`. Default is `x_range`.
    - `x_range`: Generates x values based on the range of the features.
    - `x_features`: Specify a set of x values. Default is `None`.
    - `sample`: Samples x values randomly from the dataset that are not in the context.
- `num_x_samples`: If `x_row_method` is `sample`, this is the number of x values to sample. Default is `1`.
- `x_features`: If `x_row_method` is `x_features`, this is the set of x values to use. Provide as a string of a dictionary. e.g. for x values (0.5, 0.3), and (0.3, 0.4) the input would be `"{'feature1': [0.5, 0.6], 'feature2': [0.3, 0.4]}"`. Default is `None`.
- `x_range`: If `x_row_method` is `x_range`, this is the grid of x values to use. Provide as a string of a dictionary. e.g. for a grid of x values where `feature1` is the range $[0.5, 0.6)$ with step 0.1 and `feature2` is the range [0.3, 0.4) with step 0.1, the input would be `"{'feature1': [0.5, 0.6, 0.1], 'feature2': [0.3, 0.4, 0.1]}"`. Default is `None`.
- `x_sample_seed`: The seed for sampling x values. Default is `0`.
- `decimal_places`: The number of decimal places to round the x values to. Default is `1`.

Seed Parameters
- `numpy_seed`: The seed for NumPy random number generation. Default is `0`.
- `data_split_seed`: The seed for splitting the ICL dataset. Default is `0`.
- `icl_sample_seed`: The seed for sampling from the ICL dataset. Default is `0`.
- `fixed_permutation_seed`: If `permute_context` is `0`, this seed is used for permuting the context. Default is `0`.

Permutation Related Parameters
- `num_permutations`: The number of ICL permutations to average over. Default is `5`.
- `permute_context`: If `1`, the context is permuted when sampling. If `0`, the context is not permuted. Default is `1`.

Z Parameters
- `num_z`: The number of auxiliary z values to use. Default is `15`.
- `perturb_about_x`: If `1`, the z values are perturbed about the x values. If `0`, the z values are perturbed about the mean of the ICL data. Default is `1`.
- `perturbation_std`: The amount by which the standard deviation of the Gaussian perturbations (for generating the z values) is scaled. Default is `0.1`.
- `num_bo_z`: The number of z values to use for Bayesian Optimization. The first `num_z` - `num_bo_z` z values are randomly sampled. If `0`, no Bayesian Optimization is performed. Default is `0`.
- `num_candidates`: The number of candidates to generate for Bayesian Optimization. Default is `3`.

Other parameters
- `run_name`: The name of the run. Default is `test`.
- `save_directory`: The sub-directory within `/results/toy_classification` or `/results/toy_regression` (respectively) to save the results in. Default is `other`.
- `verbose_output`: If `1`, verbose output is printed. Default is `0`.

### Bandits

Example Scripts:

```
python run_bandit_classification.py

python run_bandit_classification.py --model_temperature 2.0 --bandit_num_arms 10 --bandit_midpoint 0.6 --bandit_gap 0.1 --bandit_exploration_rate 1.0 --num_trials 100 --num_random_trials 10 --uncertainty_type total --run_name buttons_midpoint_0.6_gap_0.1 --save_directory 10_arm_bandit
```

Parameters:

API Parameters
- `model_name`: The name of the model to use for predictions. Options: `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-14B` and `meta-llama/Meta-Llama-3-8B`. `Qwen/Qwen2.5-14B` is the default.
- `model_port`: The port number for the model server. Default is `8000`.
- `model_ip`: The IP address of the model server. Default is `localhost`.
- `model_temperature`: The temperature for the model. Default is `1.0`.
- `is_local_client`: Whether to use a local client for the model. Default is `1` (True). `0` for OpenAI API.

Bandit Parameters
- `bandit_name`: Name of the bandit to be used. Default is "buttons".
- `bandit_num_arms`: Number of arms for the bandit. Default is `5`.
- `bandit_midpoint`: Midpoint reward probability for the bandit. Default is `0.5`.
- `bandit_gap`: Gap between the best and worst arm. Default is `0.2`.
- `bandit_seed`: Seed for the bandit reward generation. Default is `0`.
- `bandit_exploration_rate`: Exploration rate for the bandit algorithm. Default is `2.0`.
- `is contextual_bandit`: `0` if a contextual bandit problem. `1` otherwise. Default is `0`

Experiment Parameters
- `num_trials`: Number of trials to run. Default is `10`.
- `num_random_trials`: Number of random trials to run. Default is `3`.
- `uncertainty_type`: Type of uncertainty to use. Default is "epistemic". Options are "epistemic", "total", and "ucb1".

Seed Parameters
- `numpy_seed`: The seed for NumPy random number generation. Default is `0`.
- `fixed_permutation_seed`: If `permute_context` is `0`, this seed is used for permuting the context. Default is `0`.

Permutation Related Parameters
- `num_permutations`: The number of ICL permutations to average over. Default is `10`.
- `permute_context`: If `1`, the context is permuted when sampling. If `0`, the context is not permuted. Default is `1`.

Z Parameters
- `num_z`: The number of auxiliary z values to use. Default is `1`.
- `perturbation_std`: The amount by which the standard deviation of the Gaussian perturbations (for generating the z values) is scaled. Default is `1.0`.
- `decimal_places`: The number of decimal places to round the x values to. Default is `1`.
- `min_KL_rank`: Chooses the z value with the lowest Va from the z values with smallest `k` KL values. Default `k=1`.

Other parameters
- `run_name`: The name of the run. Default is `test`.
- `save_directory`: The sub-directory within `/results/bandits` to save the results in. Default is `other`.
- `verbose_output`: If `1`, verbose output is printed. Default is `0`.


### OOD Detection

Available built-in question-answering datasets to run:

**BoolQA**: https://arxiv.org/abs/1905.10044

**HotPotQA**: https://arxiv.org/abs/1809.09600

**PubMedQA**: https://aclanthology.org/D19-1259/

Scripts:

```
python run_qa.py --id [NAME_OF_ID_DATASET] --ood [NAME_OF_OOD_DATASET]
```
```
python run_qa.py --id boolqa --ood pubmedqa
```

Parameters:

- `id`: The name of the in-distribution dataset to use. Options: `boolqa`, `hotpotqa`, `pubmedqa`. Default is `boolqa`.
- `ood`: The name of the out-of-distribution dataset to use. Options: `boolqa`, `hotpotqa`, `pubmedqa`. Default is `pubmedqa`.
- `num_D`: Number of in-context training examples. Default is `15`.
- `num_z`: Number of z perturbations. Default is `20`.

Evaluation:

Before evaluating out-of-distribution results, ensure that the data paths are updated in `eval_ood.py`.

```
python eval_ood.py
```

## Citation

Please consider citing our paper if you find it helpful. Thank you :grinning:!

```
@misc{jayasekera2025variationaluncertaintydecompositionincontext,
      title={Variational Uncertainty Decomposition for In-Context Learning}, 
      author={I. Shavindra Jayasekera and Jacob Si and Filippo Valdettaro and Wenlong Chen and A. Aldo Faisal and Yingzhen Li},
      year={2025},
      eprint={2509.02327},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2509.02327}, 
}
```
