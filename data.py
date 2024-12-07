import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample 
import random
from torch.nn.utils import clip_grad_norm_
import time
import csv
from openai import OpenAI
import re
from scipy import sparse

from utils import *

def construct_fact_dict(fact_rdf):
    fact_dict = {}
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        if r not in fact_dict:
            fact_dict[r] = []
        fact_dict[r].append(rdf)

    return fact_dict  

def parse_rdf(rdf):
    """
        return: head, relation, tail
    """
    rdf_tail, rdf_rel, rdf_head= rdf
    return rdf_head, rdf_rel, rdf_tail
   

def sample_anchor_rdf(rdf_data, num=1):
    if num < len(rdf_data):
        return sample(rdf_data, num)
    else:
        return rdf_data

def construct_descendant(rdf_data):
    # take entity as h, map it to its r, t
    entity2desced_r = {}
    entity2desced_t = {}
    for rdf_ in rdf_data:
        h_, r_, t_ = rdf_
        if h_ not in entity2desced_r.keys():
            entity2desced_r[h_] = [r_]
            entity2desced_t[h_] = [t_]
        else:
            entity2desced_r[h_].append(r_)
            entity2desced_t[h_].append(t_)
    return entity2desced_r, entity2desced_t

    
def connected(entity2desced_r, entity2desced_t, head, tail):
    if head in entity2desced_t:
        if tail in entity2desced_t[head]:
            idx = entity2desced_t[head].index(tail)
            return entity2desced_r[head][idx]
        return False
    else:
        return False


def construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf):
    e_num = len(idx2ent)
    r2mat = {}
    # initialize rmat
    for idx, rel in idx2rel.items():
        mat = sparse.dok_matrix((e_num, e_num))
        r2mat[rel] = mat
    # fill rmat
    for rdf in fact_rdf:
        fact = rdf
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        r2mat[r][h_idx, t_idx] = 1
    return r2mat


class Dictionary(object):
    def __init__(self):
        self.rel2idx_ = {}
        self.idx2rel_ = {}
        self.idx = 0

    def add_relation(self, rel):
        if rel not in self.rel2idx_.keys():
            self.rel2idx_[rel] = self.idx
            self.idx2rel_[self.idx] = rel
            self.idx += 1

    @property
    def rel2idx(self):
        return self.rel2idx_

    @property
    def idx2rel(self):
        return self.idx2rel_

    def __len__(self):
        return len(self.idx2rel_)


def load_entities(path):
    idx2ent, ent2idx = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            e = line.strip()
            ent2idx[e] = idx
            idx2ent[idx] = e
    return idx2ent, ent2idx


class Dataset(object):
    def __init__(self, data_root, sparsity=1, inv=False):
        # Construct entity_list
        entity_path = data_root + 'entities.txt'
        self.idx2ent_, self.ent2idx_ = load_entities(entity_path)
        # Construct rdict which contains relation2idx & idx2relation2
        relation_path = data_root + 'relations.txt'
        self.rdict = Dictionary()
        self.load_relation_dict(relation_path)
        # head relation
        self.head_rdict = Dictionary()
        self.head_rdict = copy.deepcopy(self.rdict)
        # load (h, r, t) tuples
        fact_path = data_root + 'facts.txt'
        train_path = data_root + 'train.txt'
        valid_path = data_root + 'valid.txt'
        test_path = data_root + 'test.txt'
        if inv:
            fact_path += '.inv'
        self.rdf_data_ = self.load_data_(fact_path, train_path, valid_path, test_path, sparsity)
        self.fact_rdf_, self.train_rdf_, self.valid_rdf_, self.test_rdf_ = self.rdf_data_
        # inverse
        if inv:
            # add inverse relation to rdict
            rel_list = list(self.rdict.rel2idx_.keys())
            for rel in rel_list:
                inv_rel = "inv_" + rel
                self.rdict.add_relation(inv_rel)
                self.head_rdict.add_relation(inv_rel)
                # add None
        self.head_rdict.add_relation("None")

    def load_rdfs(self, path):
        rdf_list = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tuples = line.strip().split('\t')
                rdf_list.append(tuples)
        return rdf_list

    def load_data_(self, fact_path, train_path, valid_path, test_path, sparsity):
        fact = self.load_rdfs(fact_path)
        fact = sample(fact, int(len(fact) * sparsity))
        train = self.load_rdfs(train_path)
        valid = self.load_rdfs(valid_path)
        test = self.load_rdfs(test_path)
        return fact, train, valid, test

    def load_relation_dict(self, relation_path):
        with open(relation_path, encoding='utf-8') as f:
            rel_list = f.readlines()
            for r in rel_list:
                relation = r.strip()
                self.rdict.add_relation(relation)

    def get_relation_dict(self):
        return self.rdict

    def get_head_relation_dict(self):
        return self.head_rdict

    @property
    def idx2ent(self):
        return self.idx2ent_

    @property
    def ent2idx(self):
        return self.ent2idx_

    @property
    def fact_rdf(self):
        return self.fact_rdf_

    @property
    def train_rdf(self):
        return self.train_rdf_

    @property
    def valid_rdf(self):
        return self.valid_rdf_

    @property
    def test_rdf(self):
        return self.test_rdf_


def dynamic_simple(anchor_rdf, entity2desced_r, entity2desced_t, max_rule_len, simple_per_anchor, candidate_rules, temperature):
    counts = 0
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_instance_initial = anchor_t + '|' + anchor_r + '|' + anchor_h
    stack_instance = [stack_instance_initial]
    expended_node = []
    expends = [anchor_h]
    rule_seqs = []
    instance_seqs = []

    if (anchor_r in candidate_rules) and (random.random() < temperature):
        candidate_rule = candidate_rules[anchor_r]

        for current_rules in candidate_rule:
            len_current_rules = len(current_rules)
            cur_r_stack = [anchor_r]
            cur_t_stack = [anchor_t]
            cur_ins_stack = [stack_instance_initial]
            cur_r_stack_temp = []
            cur_t_stack_temp = []
            cur_ins_stack_temp = []
            rule_index = -1
            expends2 = [anchor_h]

            for rule_index in range(len_current_rules-1):
                break_flag = False
                while len(cur_t_stack) > 0:
                    cur_t = cur_t_stack.pop(-1)
                    cur_r = cur_r_stack.pop(-1)
                    cur_ins = cur_ins_stack.pop(-1)
                    r_ = current_rules[rule_index]
                    if cur_t in entity2desced_r:                
                        if r_ in entity2desced_r[cur_t]:
                            break_flag = True
                            for current_rule_idx, value in enumerate(entity2desced_r[cur_t]):
                                if value == r_:
                                    t_ = entity2desced_t[cur_t][current_rule_idx]
                                    if t_ not in expends2:
                                        cur_r_stack_temp.append(cur_r+'|' + r_)
                                        stack.append((cur_t, cur_r +'|' + r_, t_))
                                        cur_ins_stack_temp.append(cur_ins+'--' + cur_t + '|' + r_ + '|' + t_)
                                        stack_instance.append(cur_ins+'--' + cur_t + '|' + r_ + '|' + t_)
                                        cur_t_stack_temp.append(t_)
                    expends2.append(cur_t)
                cur_t_stack = cur_t_stack_temp
                cur_r_stack = cur_r_stack_temp
                cur_ins_stack = cur_ins_stack_temp
                cur_t_stack_temp = []
                cur_r_stack_temp = []
                cur_ins_stack_temp = []
                if break_flag == False:
                    break

            if rule_index == len_current_rules-2:
                rule_index += 1
                while len(cur_t_stack) > 0:
                    cur_t = cur_t_stack.pop(-1)
                    cur_r = cur_r_stack.pop(-1)
                    cur_ins = cur_ins_stack.pop(-1)
                    rule_head_rel = False
                    rule_head_rel = connected(entity2desced_r, entity2desced_t, cur_t, anchor_h)
                    if ((max_rule_len > 1) and rule_head_rel) or ((max_rule_len == 1) and rule_head_rel and (rule_head_rel != anchor_r) and (rule_head_rel.replace('inv_', '') != anchor_r) and (rule_head_rel != anchor_r.replace('inv_', ''))):
                        cur_r += '|' + rule_head_rel
                        rule_seq = cur_r
                        cur_ins += '--' + cur_t + '|' + rule_head_rel + '|' + anchor_h

                        if cur_ins not in expended_node:
                            expended_node.append(cur_ins)

                            cur_instance = cur_ins.split('--')
                            instance_seq = []
                            for cur_instance_split in cur_instance:
                                cur_instance_splits = cur_instance_split.split('|')
                                instance_seq.append(cur_instance_splits)
                            rule_seqs.append(rule_seq)
                            instance_seqs.extend([instance_seq])
                            counts += 1
                            stack.pop(-1)
                            stack_instance.pop(-1)
                            if counts > simple_per_anchor-1:
                                break


    # Search
    while len(stack) > 0 and counts < simple_per_anchor:
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_instance = stack_instance.pop(-1)
        cur_r_split = cur_r.split('|')

        if len(cur_r_split)== max_rule_len:
            rule_head_rel = False
            rule_head_rel = connected(entity2desced_r, entity2desced_t, cur_t, anchor_h)
            instance_seq = []
            if ((max_rule_len > 1) and rule_head_rel and (cur_t != anchor_t)) or ((max_rule_len == 1) and rule_head_rel and (rule_head_rel != anchor_r) and (rule_head_rel.replace('inv_', '') != anchor_r) and (rule_head_rel != anchor_r.replace('inv_', ''))):
                rule_seq = cur_r + '|' + rule_head_rel
                cur_instance += '--' + cur_t + '|' + rule_head_rel + '|' + anchor_h
                if cur_instance not in expended_node:
                    expended_node.append(cur_instance)
                    cur_instances = cur_instance.split('--')
                    for cur_instance_split in cur_instances:
                        cur_instance_splits = cur_instance_split.split('|')
                        instance_seq.append(cur_instance_splits)
                    rule_seqs.append(rule_seq)
                    instance_seqs.extend([instance_seq])
                    counts += 1
                    if counts > simple_per_anchor-1:
                        break

        elif (len(cur_r_split) < max_rule_len) and (cur_t in entity2desced_r):
            for i in range(len(entity2desced_t[cur_t])):
                t_ = entity2desced_t[cur_t][i]
                if t_ not in expends:
                    r_ = entity2desced_r[cur_t][i]
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_instance.append(cur_instance +'--' + cur_t + '|' + r_ + '|' + t_)
            expends.append(cur_t)

    return rule_seqs, instance_seqs

def evaluate_rules(rule_seqs, instance_seq, candidate_rules, candidate_rules_count, valid_rules, candidate_wrong_rules, wrong_rules, threshold_instance, threshold_valid, api_key, data_type, gpt_model):
    num_tokens = 0
    for i in range(len(rule_seqs)):
        rule_seq = rule_seqs[i]
        temp_rule = re.split(r'--|\|', rule_seq)
        rule_end = temp_rule[-1]
        if rule_end != 'None':
            temp_rule_body = temp_rule[1:]
            if rule_seq in valid_rules:
                valid_rules[rule_seq][0] += 10
            elif rule_seq in wrong_rules:
                wrong_rules[rule_seq][1] += 10
            else:
                instance_assessment, num_token = gpt(instance_seq[i], api_key, data_type, gpt_model)
                num_tokens += num_token
                if instance_assessment == 1:
                    if temp_rule[0] in candidate_rules:
                        if temp_rule_body not in candidate_rules[temp_rule[0]]:
                            candidate_rules[temp_rule[0]].extend([temp_rule_body])
                            candidate_rules_count[rule_seq] = [1, 0]
                        else:
                            candidate_rules_count[rule_seq][0] += 1
                            if (candidate_rules_count[rule_seq][0] >= threshold_instance) and (candidate_rules_count[rule_seq][0]/(candidate_rules_count[rule_seq][0]+candidate_rules_count[rule_seq][1]) >= threshold_valid):
                                valid_rules[rule_seq] = candidate_rules_count[rule_seq]
                                index = candidate_rules[temp_rule[0]].index(temp_rule_body)
                                del candidate_rules[temp_rule[0]][index]
                                del candidate_rules_count[rule_seq]
                            elif candidate_rules_count[rule_seq][0] >= 2*threshold_instance:
                                wrong_rules[rule_seq] = candidate_rules_count[rule_seq]
                                index = candidate_rules[temp_rule[0]].index(temp_rule_body)
                                del candidate_rules[temp_rule[0]][index]
                                del candidate_rules_count[rule_seq]
                    else:
                        candidate_rules[temp_rule[0]] = [temp_rule_body]
                        candidate_rules_count[rule_seq] = [1, 0]
                    if rule_seq in candidate_wrong_rules:
                        candidate_wrong_rules[rule_seq] -= 1
                        if candidate_wrong_rules[rule_seq] < 1:
                            del candidate_wrong_rules[rule_seq]                
                elif instance_assessment == 0:
                    if rule_seq in candidate_wrong_rules:
                        candidate_wrong_rules[rule_seq] += 1
                        if candidate_wrong_rules[rule_seq] >= 2 *threshold_instance:
                            wrong_rules[rule_seq] = [0, candidate_wrong_rules[rule_seq]]
                            if rule_seq in candidate_rules_count:
                                del candidate_rules_count[rule_seq]
                                index = candidate_rules[temp_rule[0]].index(temp_rule_body)
                                del candidate_rules[temp_rule[0]][index]
                            del candidate_wrong_rules[rule_seq]
                    else:
                        candidate_wrong_rules[rule_seq] = 1
                    if temp_rule[0] in candidate_rules:
                        if temp_rule_body in candidate_rules[temp_rule[0]]:
                            candidate_rules_count[rule_seq][1] += 1
                            if (candidate_rules_count[rule_seq][1] >= threshold_instance) and (candidate_rules_count[rule_seq][0] <= candidate_rules_count[rule_seq][1]):
                                wrong_rules[rule_seq] = candidate_rules_count[rule_seq]
                                index = candidate_rules[temp_rule[0]].index(temp_rule_body)
                                del candidate_rules[temp_rule[0]][index]
                                del candidate_rules_count[rule_seq]
                                if rule_seq in candidate_wrong_rules:
                                    del candidate_wrong_rules[rule_seq]
                                    
    return candidate_rules, candidate_rules_count, valid_rules, candidate_wrong_rules, wrong_rules, num_tokens

def gpt(instances, api_key, data_type, gpt_model):
    max_retries = 100
    retry_delay = 10

    head = instances.pop(0)
    length = len(instances)
 
    str1 = ''
    str2 = ''
    str3 = 'Based on the facts above, '
    str6 = 'Please answer with "Yes" or "No". Do not output other words. '

    if data_type == 'family':
        part1_1 = 'Person '
        part1_2 = ' has a relationship of "'
        part1_3 = '" with person '
        part1_4 = '. '
        part2_1 = 'can we infer that person '
        part2_2 = ' have a relationship of "'
        part2_3 = '" with person '
        part2_4 = '? '

        # Given
        for j in range(length):
            if "inv_" not in instances[j][1]:
                str1 += part1_1 + instances[j][0] + part1_2 + instances[j][1] + part1_3 + instances[j][2] + part1_4
            else:
                str1 += part1_1 + instances[j][2] + part1_2 + instances[j][1].replace('inv_', '') + part1_3 + instances[j][0] + part1_4
        # Question
        if "inv_" not in head[1]:
            str2 = part2_1 + head[0] + part2_2 + head[1] + part2_3 + head[2] + part2_4
        else:
            str2 = part2_1 + head[2] + part2_2 + head[1].replace('inv_', '') + part2_3 + head[0] + part2_4

    elif data_type == 'wn-18rr':
        part1_1 = 'Concept '
        part1_2 = ' has a relationship of "'
        part1_3 = '" with concept '
        part1_4 = '. '
        part2_1 = 'can we infer that concept '
        part2_2 = ' have a relationship of "'
        part2_3 = '" with concept '
        part2_4 = '? '

        # Given
        for j in range(length):
            if "inv_" not in instances[j][1]:
                str1 += part1_1 + instances[j][0].replace('_', ' ') + part1_2 + instances[j][1].replace('_', ' ') + part1_3 + instances[j][2].replace('_', ' ') + part1_4
            else:
                str1 += part1_1 + instances[j][2].replace('_', ' ') + part1_2 + instances[j][1].replace('inv_', '').replace('_', ' ') + part1_3 + instances[j][0].replace('_', ' ') + part1_4
        # Question
        if "inv_" not in head[1]:
            str2 = part2_1 + head[0].replace('_', ' ') + part2_2 + head[1].replace('_', ' ') + part2_3 + head[2].replace('_', ' ') + part2_4
        else:
            str2 = part2_1 + head[2].replace('_', ' ') + part2_2 + head[1].replace('inv_', '').replace('_', ' ') + part2_3 + head[0].replace('_', ' ') + part2_4

    elif data_type == 'yago':
        part1_1 = '"'
        part1_2 = '" has a relationship of "'
        part1_3 = '" with "'
        part1_4 = '". '
        part2_1_1 = 'can we infer that "'
        part2_1_2 = ''
        part2_2 = '" have a relationship of "'
        part2_3 = '" with "'
        part2_4 = '"? '

        # Given
        for j in range(length):
            if "inv_" not in instances[j][1]:
                rela = re.sub(r'([A-Z])', r' \1', instances[j][1]).lower().strip()
                str1 += part1_1 + instances[j][0].replace('_', ' ') + part1_2 + rela + part1_3 + instances[j][2].replace('_', ' ') + part1_4
            else:
                rela = re.sub(r'([A-Z])', r' \1', instances[j][1].replace('inv_', '')).lower().strip()
                str1 += part1_1 + instances[j][2].replace('_', ' ') + part1_2 + rela + part1_3 + instances[j][0].replace('_', ' ') + part1_4
        # Question
        if "inv_" not in head[1]:
            rela = re.sub(r'([A-Z])', r' \1', head[1]).lower().strip()
            str2 = part2_1_1 + part2_1_2 + head[0].replace('_', ' ') + part2_2 + rela + part2_3 + head[2].replace('_', ' ') + part2_4
        else:
            rela = re.sub(r'([A-Z])', r' \1', head[1].replace('inv_', '')).lower().strip()
            str2 = part2_1_1 + part2_1_2 + head[2].replace('_', ' ') + part2_2 + rela + part2_3 + head[0].replace('_', ' ') + part2_4

    elif data_type == 'fb15k-237':
        part1_1 = 'Entity "'
        part1_2 = '" has a relationship of "'
        part1_3 = '" with entity "'
        part1_4 = '". '
        part2_1 = 'can we infer that entity "'
        part2_2 = '" have a relationship of "'
        part2_3 = '" with entity "'
        part2_4 = '"? '

        # Given
        for j in range(length):
            if "inv_" not in instances[j][1]:
                str1 += part1_1 + instances[j][0] + part1_2 + instances[j][1] + part1_3 + instances[j][2] + part1_4
            else:
                str1 += part1_1 + instances[j][2] + part1_2 + instances[j][1].replace('inv_', '') + part1_3 + instances[j][0]+ part1_4
        # Question
        if "inv_" not in head[1]:
            str2 = part2_1 + head[0] + part2_2 + head[1] + part2_3 + head[2] + part2_4
        else:
            str2 = part2_1 + head[2] + part2_2 + head[1].replace('inv_', '') + part2_3 + head[0] + part2_4

    prompt1 = str1 + str3 + str2 + str6

    MESSAGE=[
            {"role": "system", "content": "You are an expert in the field of biochemistry."},
            {"role": "user", "content": prompt1}
            ]

    # num_token = num_tokens_from_messages(MESSAGE, gpt_model)
    num_token = 0

    for i in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model= gpt_model,
                messages=MESSAGE,
                temperature=0.5,
                max_tokens=2000,
                n=1,
                stop=None,
            )

            full_response = response
            contents = full_response.choices[0].message.content
            pass
        except Exception as e:
            print(f"Network connection interrupted.")
            if i < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                print(f"Attempting retry #{i+2}...")
            else:
                print("Maximum retry limit reached, exit program.")
                break
        else:
            break

    if (contents == 'Yes') or (contents == 'yes') or (contents == 'YES') or (contents == 'Yes.') or (contents == 'yes.') or (contents == 'YES.'):
        instance_assessment=1
    elif (contents == 'No') or (contents == 'no') or (contents == 'NO') or (contents == 'No.') or (contents == 'no.') or (contents == 'NO.'):
        instance_assessment=0
    else:
        instance_assessment=4

    return instance_assessment, num_token

def gpt_semantic_assessment(contra_list, valid_dict, api_key, data_type, gpt_model):
    valid = list(valid_dict.keys())
    rule2 = random.choice(valid)
    rule1 = random.choice(contra_list)
    while rule1 in valid:
        rule1 = random.choice(contra_list)

    rule1s = rule1.split('|')
    rule2s = rule2.split('|')
    head1 = rule1s.pop(0)
    head2 = rule2s.pop(0)
    rule1s.append(head1)
    rule2s.append(head2)

    str7 = ' are entities. Which of the following two logical inferences is more reliable? '
    str8 = ' Think step by step to assess whether these two options are reasonable reasoning. Firstly, for each option, judge whether the inference in the third sentence is reliable based on what is known in the first two sentences. Then, compare the reliability of the two options and choose the higher option. '
    str9 = 'Please answer with "Option 1" or "Option 2". '
    str10 = 'Output in JSON format, for example: {"Explanation": "", "Answer (Option 1 or Option 2)": ""} '
    
    option1 = 'Option 1:'
    option2 = ' Option 2:'

    entity = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']

    if data_type == 'family':
        part1_1 = 'Person '
        part1_2 = ' is the '
        part1_3 = ' of person '
        part1_4 = '. '
        part2_1 = 'So we can infer that person '
        part2_2 = ' is the '
        part2_3 = ' of person '
        part2_4 = '. '
    
        length1 = len(rule1s)
        str1 = option1
        for j in range(length1-1):
            if 'inv_' not in rule1s[j]:
                str1 += part1_1 + entity[j] + part1_2 + rule1s[j] + part1_3 + entity[j+1] + part1_4
            else:
                str1 += part1_1 + entity[j+1] + part1_2 + rule1s[j].replace('inv_', '') + part1_3 + entity[j] + part1_4
        if 'inv_' not in rule1s[-1]:
            str2 = part2_1 + entity[0] + part2_2 + rule1s[-1] + part2_3 + entity[j+1] + part2_4
        else:
            str2 = part2_1 + entity[j+1] + part2_2 + rule1s[-1].replace('inv_', '') + part2_3 + entity[0] + part2_4 

        length2 = len(rule2s)
        str3 = option2
        j = j+2
        for k in range(length2-1):
            if 'inv_' not in rule2s[k]:
                str3 += part1_1 + entity[j+k] + part1_2 + rule2s[k] + part1_3 + entity[j+k+1] + part1_4
            else:
                str3 += part1_1 + entity[j+k+1] + part1_2 + rule2s[k].replace('inv_', '') + part1_3 + entity[j+k] + part1_4
        if 'inv_' not in rule2s[-1]:
            str4 = part2_1 + entity[j] + part2_2 + rule2s[-1] + part2_3 + entity[j+k+1] + part2_4 
        else:
            str4 = part2_1 + entity[j+k+1] + part2_2 + rule2s[-1].replace('inv_', '') + part2_3 + entity[j] + part2_4 
    
    elif data_type == 'wn-18rr':
        part1_1 = 'Concept '
        part1_2 = ' has a relationship of "'
        part1_3 = '" with concept '
        part1_4 = '. '
        part2_1 = 'So we can infer that concept '
        part2_2 = ' have a relationship of "'
        part2_3 = '" with concept '
        part2_4 = '. '

        length1 = len(rule1s)
        str1 = option1
        for j in range(length1-1):
            if 'inv_' not in rule1s[j]:
                str1 += part1_1 + entity[j] + part1_2 + rule1s[j].replace('_', ' ') + part1_3 + entity[j+1] + part1_4
            else:
                str1 += part1_1 + entity[j+1] + part1_2 + rule1s[j].replace('inv_', '').replace('_', ' ') + part1_3 + entity[j] + part1_4
        if 'inv_' not in rule1s[-1]:
            str2 = part2_1 + entity[0] + part2_2 + rule1s[-1].replace('_', ' ') + part2_3 + entity[j+1] + part2_4
        else:
            str2 = part2_1 + entity[j+1] + part2_2 + rule1s[-1].replace('inv_', '').replace('_', ' ') + part2_3 + entity[0] + part2_4 

        length2 = len(rule2s)
        str3 = option2
        j = j+2
        for k in range(length2-1):
            if 'inv_' not in rule2s[k]:
                str3 += part1_1 + entity[j+k] + part1_2 + rule2s[k].replace('_', ' ') + part1_3 + entity[j+k+1] + part1_4
            else:
                str3 += part1_1 + entity[j+k+1] + part1_2 + rule2s[k].replace('inv_', '').replace('_', ' ') + part1_3 + entity[j+k] + part1_4
        if 'inv_' not in rule2s[-1]:
            str4 = part2_1 + entity[j] + part2_2 + rule2s[-1].replace('_', ' ') + part2_3 + entity[j+k+1] + part2_4 
        else:
            str4 = part2_1 + entity[j+k+1] + part2_2 + rule2s[-1].replace('inv_', '').replace('_', ' ') + part2_3 + entity[j] + part2_4 


    elif data_type == 'fb15k-237':
        part1_1 = 'Entity "'
        part1_2 = '" has a relationship of "'
        part1_3 = '" with entity "'
        part1_4 = '". '
        part2_1 = 'So we can infer that entity "'
        part2_2 = '" have a relationship of "'
        part2_3 = '" with entity "'
        part2_4 = '". '

        length1 = len(rule1s)
        str1 = option1
        for j in range(length1-1):
            if 'inv_' not in rule1s[j]:
                str1 += part1_1 + entity[j] + part1_2 + rule1s[j] + part1_3 + entity[j+1] + part1_4
            else:
                str1 += part1_1 + entity[j+1] + part1_2 + rule1s[j].replace('inv_', '') + part1_3 + entity[j] + part1_4
        if 'inv_' not in rule1s[-1]:
            str2 = part2_1 + entity[0] + part2_2 + rule1s[-1] + part2_3 + entity[j+1] + part2_4
        else:
            str2 = part2_1 + entity[j+1] + part2_2 + rule1s[-1].replace('inv_', '') + part2_3 + entity[0] + part2_4 

        length2 = len(rule2s)
        str3 = option2
        j = j+2
        for k in range(length2-1):
            if 'inv_' not in rule2s[k]:
                str3 += part1_1 + entity[j+k] + part1_2 + rule2s[k] + part1_3 + entity[j+k+1] + part1_4
            else:
                str3 += part1_1 + entity[j+k+1] + part1_2 + rule2s[k].replace('inv_', '') + part1_3 + entity[j+k] + part1_4
        if 'inv_' not in rule2s[-1]:
            str4 = part2_1 + entity[j] + part2_2 + rule2s[-1] + part2_3 + entity[j+k+1] + part2_4 
        else:
            str4 = part2_1 + entity[j+k+1] + part2_2 + rule2s[-1].replace('inv_', '') + part2_3 + entity[j] + part2_4 


    elif data_type == 'yago':
        part1_1 = ''
        part1_2 = ' '
        part1_3 = ' '
        part1_4 = '. '
        part2_1 = 'So we can infer that '
        part2_2 = ' '
        part2_3 = ' '
        part2_4 = '. '

        length1 = len(rule1s)
        str1 = option1
        for j in range(length1-1):
            if 'inv_' not in rule1s[j]:
                rela = re.sub(r'([A-Z])', r' \1', rule1s[j]).lower().strip()
                str1 += part1_1 + entity[j] + part1_2 + rela + part1_3 + entity[j+1] + part1_4
            else:
                rela = re.sub(r'([A-Z])', r' \1', rule1s[j].replace('inv_', '')).lower().strip()
                str1 += part1_1 + entity[j+1] + part1_2 + rela + part1_3 + entity[j] + part1_4
        if 'inv_' not in rule1s[-1]:
            rela = re.sub(r'([A-Z])', r' \1', rule1s[-1]).lower().strip()
            str2 = part2_1 + entity[0] + part2_2 + rela + part2_3 + entity[j+1] + part2_4
        else:
            rela = re.sub(r'([A-Z])', r' \1', rule1s[-1].replace('inv_', '')).lower().strip()
            str2 = part2_1 + entity[j+1] + part2_2 + rela + part2_3 + entity[0] + part2_4 

        length2 = len(rule2s)
        str3 = option2
        j = j+2
        for k in range(length2-1):
            if 'inv_' not in rule2s[k]:
                rela = re.sub(r'([A-Z])', r' \1', rule2s[k]).lower().strip()
                str3 += part1_1 + entity[j+k] + part1_2 + rela + part1_3 + entity[j+k+1] + part1_4
            else:
                rela = re.sub(r'([A-Z])', r' \1', rule2s[k].replace('inv_', '')).lower().strip()
                str3 += part1_1 + entity[j+k+1] + part1_2 + rela + part1_3 + entity[j+k] + part1_4
        if 'inv_' not in rule2s[-1]:
            rela = re.sub(r'([A-Z])', r' \1', rule2s[-1]).lower().strip()
            str4 = part2_1 + entity[j] + part2_2 + rela + part2_3 + entity[j+k+1] + part2_4 
        else:
            rela = re.sub(r'([A-Z])', r' \1', rule2s[-1].replace('inv_', '')).lower().strip()
            str4 = part2_1 + entity[j+k+1] + part2_2 + rela + part2_3 + entity[j] + part2_4 


    str5 = ''
    for i in range(j+k+1):
        str5 += entity[i] + ', '
    str5 += entity[j+k+1] + str7

    prompt1 = str5 + str1 + str2 + str3 + str4 + str8 + str9 + str10

    MESSAGE=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt1}
            ]

    max_retries = 100
    retry_delay = 10
    for i in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model= gpt_model,
                messages=MESSAGE,
                temperature=0,
                max_tokens=2000,
                n=1,
                stop=None,
                response_format= {'type': "json_object"}
            )

            full_response = response
            contents = full_response.choices[0].message.content
            pass
        except Exception as e:
            print(f"Network connection interrupted.")
            if i < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                print(f"Attempting retry #{i+2}...")
            else:
                print("Maximum retry limit reached, exit program.")
                break
        else:
            break

    return contents

def sample_anchors(fact_rdf, anchor_num):
    anchors_rdf = []
    entity2desced_r, entity2desced_t = construct_descendant(fact_rdf)
    fact_dict = construct_fact_dict(fact_rdf)
    train_relation_num = len(fact_dict)
    per_anchor_num = int(anchor_num / train_relation_num)
    print("Number of per_anchor_num:{}".format(per_anchor_num))
    print("Number of train_relation:{}".format(train_relation_num))

    save_relation = {}
    for head in fact_dict.keys():
        if (head != "None") and ("inv_" not in head):
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=per_anchor_num)
            anchors_rdf.extend(sampled_rdf)
            save_relation[head] = len(sampled_rdf)
    print("Total_anchor_num",len(anchors_rdf))
    print("num_per_relation:", save_relation)

    return anchors_rdf, entity2desced_r, entity2desced_t, save_relation

def load_data(data_type):
    data_path = 'datasets/' + data_type + '/'
    if data_type == 'yago':
        file_encoding2 = 'utf-8'
    else:
        file_encoding2 = 'gbk'

    fact_rdf = []
    with open(data_path+'facts.txt.inv', 'r', encoding = file_encoding2) as f:
        lines = csv.reader(f, delimiter='\n')
        for row in lines:
            readers = csv.reader(row, delimiter='\t')
            for reader in readers:
                fact_rdf.append(reader)
    return fact_rdf


def load_valid_dict(position, max_rule_len):
    valid_dict = {}
    for lenth in range(max_rule_len, 0, -1):
        valid_dict[lenth] = {}
    with open(position, 'r') as f7:
        for line in f7:
            line = line.strip()
            ele1 = line.split('\t')
            ele2 = ele1[1].split(' <-- ')
            ele3 = ele2[1].split(', ')
            rule = ele2[0]
            len_rule = len(ele3)
            for i in range(len_rule):
                rule += '|' + ele3[i]
            ele4 = ele1[0].split(' (')
            valid_dict[len_rule][rule] = float(ele4[0])
    return valid_dict


def split_rule(rule):
    rules = rule.split('|')
    body = ''
    for i in range(1, len(rules)):
        body += rules[i] + '|'
    return body

def construct_rule_dict(data):
    rule_dict = {}
    for rule in data:
        body = split_rule(rule)
        if body not in rule_dict:
            rule_dict[body] = 0
        rule_dict[body] += data[rule][0]//10 + 1
    return rule_dict

def score_rule(rule, data, total_dict, rate):
    valid = data[rule][0] // 10 + 1
    gpt_t = data[rule][0] % 10
    gpt_f = data[rule][1]
    body = split_rule(rule)

    if gpt_t + gpt_f == 0:
        return 0

    commonscore = gpt_t / (gpt_t + gpt_f)
    structuredscore = valid / total_dict[body]
    score = rate * commonscore + (1 - rate) * structuredscore

    return score

