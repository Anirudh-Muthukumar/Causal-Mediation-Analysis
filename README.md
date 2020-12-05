## Mediation Analysis

This repository contains the code to replicate the experiments for the CS685 Project. Part of this work was adopted from paper [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/abs/2004.12265).

### Neuron Experiments

#### Create Analysis CSVs

You can run all the experiments for a given model by running the `run_profession_neuron_experiments.py` script. Just set the `-model` flag to the version you want to use and point `-out_dir` to the base directory for your results. The resulting CSV's will be saved in `${out_dir}/results/${expt}_neuron_intervention`.

Other arguments for the script:
- itype : intervention type (man_direct, woman_direct, etc)
- template : select particular template out of 8 templates
- gender : specify male, female or both
- race : specifies the ethinicity of the names used in experiment 4
- k : top k names from different ethnicity

#### Compute total effect and correlation with professions

We provide two scripts `compute_neuron_split_total_effect` and `compute_neuron_total_effect` that will report the total effects for a model in multiple different ways.

`compute_neural_total_effect` will additionally compute the correlational value between effect sizes and the bias value of the profession and generate a plot in `${out_dir}/neuron_profession_correlation.pdf`.

#### Compute aggregate neuron effects

If you want to compute the aggregate effect for each neuron, you can run `compute_and_save_neuron_agg_effect.py`, which will create a new file in `results/${date}_neuron_intervention` called `${model_name}_neuron_effects.csv` with the results.

After you have run this for each of the models you want to investigate, you can run `compute_neuron_effect_per_layer.py` which will generate plots of the per-layer effects.
One aggregate plot will be at `${out_dir}/neuron_layer_effect.pdf` and a separate plot for each model will be saved at `${out_dir}/neuron_layer_effect_${model_name}.pdf`.

#### Plotting mitigated biases for different ethinicity

We provide a script `plot_neuron_effect_name` that will report the mitigated biases for different ethinicity across layer. This in turn will generate a bar plot and a scatter plot in the same directory.

To run any experiment:
1. First run the script `run_profession_neuron_experiments.py` with required arguments which creates a bunch of `.csv` files in the `results/` directory. 
2. To generate the plots and get results for Total Effect the run the following scripts:
    - `compute_and_save_neuron_agg_effect.py`
    - `compute_neuron_split_total_effect`
    - `compute_neuron_total_effect`
