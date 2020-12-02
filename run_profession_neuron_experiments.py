"""Run all the extraction for a model across many templates. 
    Models: gpt2, bert-base-cased, roberta-base
"""
import argparse
import os
from datetime import datetime
import pandas as pd

import torch
from torch import mode
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer

from experiment import Intervention, Model
from utils import convert_results_to_pd



bert_models = ["bert-base-cased", "bert-base-uncased"]
roberta_models = ["roberta-base", "roberta-base-openai-detector", "distilroberta-base"]
gpt2_models =["gpt2", "gpt2-medium", "distilgpt2"]


parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
    "-model",
    type=str,
    default="bert-base-cased",               # changed
    help="""Model type [bert-base-cased, roberta-base, etc.].""",
)

parser.add_argument(
    "-itype",
    type=str,
    default="all",               # changed
    help="Intervention type",
)

parser.add_argument(
    "-out_dir", default=".", type=str, help="""Path of the result folder."""
)

parser.add_argument(
    "-template_indices",
    nargs="+",
    type=int,
    help="Give the indices of templates if you want to run on only a subset",
)

parser.add_argument(
    "--randomize", default=False, action="store_true", help="Randomize model weights."
)

parser.add_argument(
    "-template",
    type=str,
    default="1",               # changed
    help="Template number [\"1\", \"2\"] type",
)

parser.add_argument(
    "-gender",
    type=str,
    default="both",               # changed
    help="Gender ['male', 'female']",
)

parser.add_argument(
    "-race",
    type=str,
    default="pctaian_female",               # changed
    help="Race",
)

parser.add_argument(
    "-k",
    type=str,
    default="5",               # changed
    help="Race",
)

opt = parser.parse_args()


def get_profession_list():
    # Get the list of all considered professions
    word_list = []
    profession_set = set(["advocate", "baker", "clerk", "counselor",
"dancer", "educator", "instructor", "maid", "nurse", "planner", "poet","psychiatrist",
"secretary", "singer", "teacher", "therapist", 
    "acquaintance", "character", "citizen", "correspondent", 
"employee", "musician", "novelist", "psychologist", "student", "writer", 
"accountant", "administrator", "adventurer", "ambassador", "banker", "bartender", "chef", "composer", "critic",
"dean", "dentist", "editor", "entrepreneur", "philosopher", "physician", "trader"])

    with open("experiment_data/professions.json", "r") as f:
        for l in f:
            # there is only one line that eval"s to an array
            for j in eval(l):
                if j[0] in profession_set:
                    word_list.append(j[0])
    return word_list



def get_template_list(indices=None):

    name_list = []

    if opt.gender == "female":
        with open("experiment_data/female_names.txt", "r") as file:
            for name in file:
                if name[-1:] == '\n':
                    name_list.append(name[:-1])
                else:
                    name_list.append(name)

    elif opt.gender == "male": 
        with open("experiment_data/male_names.txt", "r") as file:
            for name in file:
                if name[-1:] == '\n':
                    name_list.append(name[:-1])
                else:
                    name_list.append(name)

    elif opt.gender == "both":      # Experiment 4
        file_name = "experiment_data/" + opt.race + "_female.csv"
        df = pd.read_csv(file_name, header = None)
        name_list += df[0][:int(opt.k)].tolist()
        
        file_name = "experiment_data/" + opt.race + "_male.csv"
        df = pd.read_csv(file_name, header = None)
        name_list += df[0][:int(opt.k)].tolist()
    
    else:
        print("Please enter a valid gender!!")
        exit(0)


    template = [
        "The {} said that",
        "The {} yelled that",
        "The {} whispered that",
        "The {} wanted that",
        "The {} desired that",
        "The {} wished that",
        "The {} ate because",
        "The {} ran because",
    ]

    templates = []

    for name in name_list:
        for i in range(len(template)):
            templates.append(name + " is a {}. " + template[i])
    

    if opt.model in bert_models + roberta_models:
        for i in range(len(templates)):
            templates[i] = templates[i] + " [MASK] is tired"

    if indices:
        subset_templates = [templates[i - 1] for i in indices]
        print("subset of templates:", subset_templates)
        return subset_templates

    return templates


def get_intervention_types():
    if opt.itype == "all":
        return ["man_direct", "man_indirect", "woman_direct", "woman_indirect"]

    return [ opt.itype ]


def construct_interventions(base_sent, professions, tokenizer, DEVICE):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    for p in professions:
        all_word_count += 1
        try:
            interventions[p] = Intervention(
                tokenizer, base_sent, [p, "man", "woman"], ["he", "she"], device=DEVICE
            )
            used_word_count += 1
        except:
            pass
    print(
        "\t Only used {}/{} professions due to tokenizer".format(
            used_word_count, all_word_count
        )
    )
    return interventions


def run_all(
    model_type="bert-base-cased",
    device="cuda",
    out_dir=".",
    random_weights=False,
    template_indices=None,
):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations.
    professions = get_profession_list()

    templates = get_template_list(template_indices)

    intervention_types = get_intervention_types()

    tokenizer_used = None

    # Initialize Model and Tokenizer.
    if opt.model in bert_models:
        tokenizer_used = BertTokenizer
    elif opt.model in roberta_models:
        tokenizer_used = RobertaTokenizer
    elif opt.model in gpt2_models:
        tokenizer_used = GPT2Tokenizer

    tokenizer = tokenizer_used.from_pretrained(model_type)
    model = Model(device=device, model=model_type, random_weights=random_weights)

    is_expt4 = False 
    base_path = None 

    # Set up folder if it does not exist.
    if opt.itype != "all":
        folder_name = "expt3_neuron_intervention_" + opt.model + "_" + opt.gender
        base_path = os.path.join(out_dir, "results", folder_name)
    else:
        is_expt4 = True 
        base_path = os.path.join(out_dir, "results", "expt4_neuron_intervention")

    if random_weights:
        base_path = os.path.join(base_path, "random")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    root = base_path 

    for temp in templates:
        folder_name = None
        if is_expt4:
            folder_name = temp.split()[0] + "_" + opt.model
            base_path = os.path.join(root, folder_name)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        
        print("Running template '{}' now...".format(temp), flush=True)
        # Fill in all professions into current template
        interventions = construct_interventions(temp, professions, tokenizer, device)
        # Consider all the intervention types
        for itype in intervention_types:
            print("\t Running with intervention: {}".format(itype), flush=True)
            # Run actual exp.
            intervention_results = model.neuron_intervention_experiment(
                interventions, itype, alpha=1.0
            )

            df = convert_results_to_pd(interventions, intervention_results)
            temp_string = "_".join(temp.replace("{}", "X").split())
            model_type_string = model_type
            fname = "_".join([temp_string, itype, model_type_string])
            df.to_csv(os.path.join(base_path, fname + ".csv"))



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_all(
        opt.model,
        device,
        opt.out_dir,
        random_weights=opt.randomize,
        template_indices=opt.template_indices,
    )
