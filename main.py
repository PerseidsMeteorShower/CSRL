import torch
import random
import argparse
import json
import json
from tqdm import tqdm
from data import *
from utils import *

random.seed(2)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print_msg(str(device))



def train(api_key):
    print_msg('  Loading data  ')
    fact_rdf = load_data(args.data_type)

    print("Sampling anchors...")
    anchors_rdf, entity2desced_r, entity2desced_t, save_relation = sample_anchors(fact_rdf, args.anchor_num)
    with open(args.result_path + '/anchor.json', 'w', encoding = args.file_encoding) as f:
        json.dump(save_relation, f)
    print("Anchors saved to anchor.json")

    print_msg("  Start learning  ")
    total_valid_rules, total_wrong_rules = {},{}
    num_tokens_total = 0

    for rule_lenth in range(args.max_rule_len, 0, -1):
        print("Learning rule lenth:", rule_lenth)
        candidate_rules, candidate_rules_count, valid_rules, candidate_wrong_rules, wrong_rules = {},{},{},{},{}
        random.shuffle(anchors_rdf)
        for anchor_rdf in tqdm(anchors_rdf, desc="Training Progress", unit="anchor"):
            rule_seq, instance_seq = dynamic_simple(anchor_rdf, entity2desced_r, entity2desced_t, rule_lenth, args.simple_per_anchor, candidate_rules, args.temperature)
            candidate_rules, candidate_rules_count, valid_rules, candidate_wrong_rules, wrong_rules, num_tokens = evaluate_rules(rule_seq, instance_seq, candidate_rules, candidate_rules_count, valid_rules, candidate_wrong_rules, wrong_rules, args.threshold_instance, args.threshold_confidence, api_key, args.data_type, args.gpt_model)
            num_tokens_total += num_tokens
        total_valid_rules.update(valid_rules)
        total_wrong_rules.update(wrong_rules)
    print('Total token number:', num_tokens_total)

    #Save results
    total_dict = construct_rule_dict(total_valid_rules)
    with open(args.result_path + '/valid_rules.txt', 'w', encoding = args.file_encoding) as f:
        for key in total_valid_rules:
            bodys = key.split('|')
            head = bodys.pop(0)
            score = score_rule(key, total_valid_rules, total_dict, args.common_ratio)
            rule = "{:.3f}\t{} <-- ".format(score, head)
            rule += ", ".join(bodys)
            f.write(rule + '\n')
    print("Valid rules saved to valid_rules.json")

    with open(args.result_path + '/wrong_rules.txt', 'w', encoding = args.file_encoding) as f:
        for key,value in total_wrong_rules.items():
            bodys = key.split('|')
            head = bodys.pop(0)
            rule = "{:.3f}\t{} <-- ".format(-value[1], head)
            rule += ", ".join(bodys)
            f.write(rule + '\n')
    print("Wrong rules saved to wrong_rules.json")

    print_msg("  End learning  ")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with given parameters.")
    parser.add_argument('--data_type', type=str, choices=['family', 'yago', 'fb15k-237', 'wn-18rr'], default='family', help='Data type')
    parser.add_argument('--result_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help='GPT model')
    parser.add_argument('--anchor_num', type=int, default=1000, help='Number of anchors')
    parser.add_argument('--max_rule_len', type=int, default=3, help='Maximum rule length')
    parser.add_argument('--simple_per_anchor', type=int, default=50, help='Simple number per anchor')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature to sample by candidate rules (against simpling randomly)')
    parser.add_argument('--threshold_instance', type=int, default=10, help='After accumulate how many instances could a rule be considered as fully evaluated')
    parser.add_argument('--threshold_confidence', type=float, default=0.8, help='Commonsense confidence to accept a rule as valid')
    parser.add_argument('--file_encoding', type=str, default='gbk', help='Output file encoding')
    parser.add_argument('--common_ratio', type=float, default=0.5, help='Ratio of commonsense confidence in final score')
    parser.add_argument('--config_file', type=str, default='config', help='Name of config file')

    global args
    args = parser.parse_args()

    #Please replace the api_key with your own api_key in the config file
    config_file = args.config_file + ".json"
    with open(config_file) as f1:
        config = json.load(f1)
        api_key = config["openai_api_key"]

    train(api_key)





