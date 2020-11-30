"""Performs attention intervention on Winogender samples and saves results to JSON file."""

import json

import fire
from pandas import DataFrame
from transformers import GPT2Tokenizer, BertTokenizer, RobertaTokenizer

import winogender
from attention_utils import perform_interventions, get_odds_ratio
from experiment import Model


def get_interventions_winogender(model_used, do_filter, stat, model, tokenizer,
                                device='cuda', filter_quantile=0.25):
    examples = winogender.load_examples()
    
    json_data = {'model_version': model_used,
            'do_filter': do_filter,
            'stat': stat,
            'num_examples_loaded': len(examples)}
    if do_filter:
        interventions = [ex.to_intervention(tokenizer, stat) for ex in examples]
        df = DataFrame({'odds_ratio': [get_odds_ratio(intervention, model) for intervention in interventions]})
        df_expected = df[df.odds_ratio > 1]
        threshold = df_expected.odds_ratio.quantile(filter_quantile)
        filtered_examples = []
        assert len(examples) == len(df)
        for i in range(len(examples)):
            ex = examples[i]
            odds_ratio = df.iloc[i].odds_ratio
            if odds_ratio > threshold:
                filtered_examples.append(ex)

        print(f'Num examples with odds ratio > 1: {len(df_expected)} / {len(examples)}')
        print(f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_examples)} / {len(examples)}')
        json_data['num_examples_aligned'] = len(df_expected)
        json_data['filter_quantile'] = filter_quantile
        json_data['threshold'] = threshold
        examples = filtered_examples
    json_data['num_examples_analyzed'] = len(examples)
    interventions = [ex.to_intervention(tokenizer, stat) for ex in examples]
    return interventions, json_data

def intervene_attention(model_used, do_filter, stat, device='cuda', filter_quantile=0.25, random_weights=False):
    model = Model(output_attentions=True, model=model_used, device=device, random_weights=random_weights)

    # Initialize Model and Tokenizer.
    if model_used == "bert-base-cased":
        tokenizer_used = BertTokenizer
    elif model_used == "roberta-base":
        tokenizer_used = RobertaTokenizer
    elif model_used == "gpt2":
        tokenizer_used = GPT2Tokenizer

    tokenizer = tokenizer_used.from_pretrained(model_used)

    interventions, json_data = get_interventions_winogender(model_used, do_filter, stat, model, tokenizer,
                                                            device, filter_quantile)
    results = perform_interventions(interventions, model)
    json_data['mean_total_effect'] = DataFrame(results).total_effect.mean()
    json_data['mean_model_indirect_effect'] = DataFrame(results).indirect_effect_model.mean()
    json_data['mean_model_direct_effect'] = DataFrame(results).direct_effect_model.mean()
    filter_name = 'filtered' if do_filter else 'unfiltered'
    if random_weights:
        model_used += '_random'
    fname = f"winogender_data/attention_intervention_{stat}_{model_used}_{filter_name}.json"
    json_data['results'] = results
    with open(fname, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    fire.Fire(intervene_attention)

