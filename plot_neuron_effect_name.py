"""
Plot the mitigated bias for each ethinicity

Usage:
    python plot_neuron_effect_name.py $result_folder_path $model_name
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd



def get_top_perc_per_layer(df, n=10):
    """Get avg indirect effect of top n% neurons"""
    num_neurons = int(df.groupby("layer_direct_mean").size()[0] * n / 100.0)
    return (
        df.groupby("layer_direct_mean")
        .apply(lambda x: x.nlargest(num_neurons, ["odds_ratio_indirect_mean"]))
        .reset_index(drop=True)
        .groupby("layer_direct_mean")
        .agg("mean")[["odds_ratio_indirect_mean", "odds_ratio_indirect_std"]]
        .reset_index()
    )


def load_files(folder_name, model_name):
    neuron_effect_fnames = []
    racenames = [f
                 for f in os.listdir(os.path.join(folder_name, model_name))
                 ]
    folder_names = {}
    for race in racenames:
        fnames = [
            f
            for f in os.listdir(os.path.join(folder_name, model_name, race))
            if ".csv" not in f
        ]
        folder_names[race] = {}

        for fname in fnames:
            csv_files = [
                f
                for f in os.listdir(os.path.join(folder_name, model_name, race))
                if fname in f and "neuron_effects" in f and f.endswith("csv")
            ]
            folder_names[race][fname] = csv_files
            neuron_effect_fnames.extend(csv_files)
    return folder_names, neuron_effect_fnames

def get_experiment2_neuron_effects(folder_path, model_name):
    neuron_effect_fnames = [f for f in os.listdir(folder_path) if "neuron_effects" in f]
    model_layer_effects = None
    for f in neuron_effect_fnames:
        file_modelname = f.split("_")[0]
        if file_modelname == model_name:
            path = os.path.join(folder_path, f)
            model_layer_effects = pd.read_csv(path)

    return get_top_perc_per_layer(model_layer_effects, 2.5)

def main(folder_name="results/", model_name="distilgpt2"):
    expt4_folder = folder_name+'experiment4/'
    expt2_folder = folder_name+'experiment2/plot_files/'

    data, neuron_effect_fnames = load_files(expt4_folder, model_name)

    name_to_effects = {}
    name_race_details = {}
    for f in neuron_effect_fnames:
        model_name, race, name = f.split("_")[:3]
        path = os.path.join(expt4_folder, model_name, race, f)
        name_to_effects[name] = pd.read_csv(path)
        name_race_details[name] = race


    # store the indirect effect by layer per name
    indirect_effect_layer_name = {}
    for name in name_to_effects:
        v = name_to_effects[name]
        indirect_effect_layer_name[name] = get_top_perc_per_layer(v, 2.5)

    experiment2_effect_layer = get_experiment2_neuron_effects(expt2_folder, model_name)

    # store the indirect effect by layer per race
    diff_effect_layer = {}

    for name, model_effect_layer  in indirect_effect_layer_name.items():
        race = name_race_details[name]
        if race not in diff_effect_layer:
            diff_effect_layer[race] =pd.DataFrame(experiment2_effect_layer['layer_direct_mean'])

        diff_effect_layer[race][name] = experiment2_effect_layer["odds_ratio_indirect_mean"].\
                                                              subtract(model_effect_layer["odds_ratio_indirect_mean"])


    cmap = {"pctblack": "#9b59b6", "pcthispanic": "#3498db", "White":"#95a5a6", "Asian Pacific Islander":"#e74c3c","Asian": "#34495e"}
    nlayers = experiment2_effect_layer.shape[0]

    for layerid in range(nlayers):
        for race in diff_effect_layer:
            color = cmap[race]
            # Ignore first column in the df (layerids)
            x = y = diff_effect_layer[race].iloc[layerid].values[1:]
            plt.scatter(x, y, color = color, label = race)
            plt.xlabel("Mitigated bias")
            plt.ylabel("Mitigated bias")
            plt.tight_layout()
            plt.legend()
            plt.savefig(
                os.path.join(expt4_folder, "neuron_layer_effect_layer" + str(layerid) + ".pdf"),
                format="pdf",
                bbox_inches="tight",
            )
        plt.close()

    print("Success, all figures were written.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<folder_name> <model_name>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    model_name =  sys.argv[2]

    main(folder_name, model_name)
