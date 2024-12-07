import random
import argparse
import json
import json
from tqdm import tqdm
from utils import *
from data import *

random.seed(2) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default="family")
    parser.add_argument("-p", default='results', help="rule path")
    parser.add_argument('--gpt_model', type=str, default='gpt-4o', help='GPT model')
    parser.add_argument('--anchor_num', type=int, default=1000, help='Number of Anchors')
    parser.add_argument('--simple_per_anchor', type=int, default=50, help='Simple number per anchor')
    parser.add_argument('--max_rule_len', type=int, default=3, help='Maximum rule body length')
    parser.add_argument('--eva_num', type=int, default=70, help='Number of rules to be tested of each length')
    parser.add_argument('--config_file', type=str, default='config', help='Name of config file')
    args = parser.parse_args()

    #Please replace the api_key with your own api_key in the config file
    config_file = args.config_file + ".json"
    with open(config_file) as f1:
        config = json.load(f1)
        api_key = config["openai_api_key"]

    print_msg('  Semantic assessment  ')
    valid_dict = load_valid_dict(args.p + '/valid_rules.txt', args.max_rule_len)

    print('Sampling random paths for comparison...')
    fact_rdf = load_data(args.data_type)
    anchors_rdf, entity2desced_r, entity2desced_t, save_relation = sample_anchors(fact_rdf, args.anchor_num)
    random_paths= {}

    for path_lenth in range(1, args.max_rule_len + 1):
        print("Sampling path lenth:", path_lenth)
        random_paths[path_lenth] = []
        for anchor_rdf in anchors_rdf:
            rule_seq, instance_seq = dynamic_simple(anchor_rdf, entity2desced_r, entity2desced_t, path_lenth, args.simple_per_anchor, {}, 0)
            random_paths[path_lenth].extend(rule_seq)
        list(set(random_paths[path_lenth]))

    print('Start LLM based assessment...')
    llm_choices, llm_answers, ture_choices = [], [], []
    for rule_lenth in range(1, args.max_rule_len + 1):
        print("Evaluating rule lenth:", rule_lenth)
        if len(valid_dict[rule_lenth]) == 0:
            continue
        for _ in tqdm(range(args.eva_num), desc="Evaluating Progress", unit="rule"):
            contents= gpt_semantic_assessment(random_paths[rule_lenth], valid_dict[rule_lenth], api_key, args.data_type, args.gpt_model)
            ture_choices.append(2)
            try:
                data = json.loads(contents)
                if "Answer (Option 1 or Option 2)" in data:
                    llm_answers.append(data["Answer (Option 1 or Option 2)"])
                    if data["Answer (Option 1 or Option 2)"] == 'Option 1':
                        llm_choices.append(1)
                    elif data["Answer (Option 1 or Option 2)"] == 'Option 2':
                        llm_choices.append(2)
                    else:
                        llm_choices.append(4)
                        print('Irregular output:', data["Answer (Option 1 or Option 2)"])
                else:
                    llm_answers.append('No_answer')
                    llm_choices.append(5)
                    print('Irregular output: No answer')
            except Exception as e:
                llm_answers.append('No_answer')
                llm_choices.append(5)
                print('Irregular output: No answer')
                continue

    equal_elements = [a == b for a, b in zip(ture_choices, llm_choices)]
    equal_count = equal_elements.count(True)
    accuracy = equal_count / len(llm_choices) * 100
    print('Accuracy (percentage):', accuracy)

    result_dict = {"Accuracy (percentage)": accuracy}
    with open(args.p + '/semantic_assessment_results.json', 'w', encoding = 'utf-8') as f:
        json.dump(result_dict, f)





