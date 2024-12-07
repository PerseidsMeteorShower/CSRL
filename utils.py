import tiktoken
import numpy as np

def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg) 
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")


def num_tokens_from_messages(messages, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


def ill_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def harsh_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth]
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank

def balance_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    if pred[ent2idx[q_t]]!=0:
        pred_ranks = np.argsort(pred)[::-1]    

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t!=ent2idx[q_t]]

        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth:
                filtered_ranks.append(idx)

        rank = filtered_ranks.index(ent2idx[q_t])+1
    else:
        truth = gt[(q_h, q_r)]

        filtered_pred = []

        for i in range(len(pred)):
            if i not in truth:
                filtered_pred.append(pred[i])
        n_non_zero = np.count_nonzero(filtered_pred)
        rank = n_non_zero+1
    return rank

def random_rank(pred, gt, ent2idx, q_h, q_t, q_r):
    pred_ranks = np.argsort(pred)[::-1]
    truth = gt[(q_h, q_r)]
    truth = [t for t in truth if t != ent2idx[q_t]]
    truth.append(ent2idx[q_t])
    filtered_ranks = []
    for i in range(len(pred_ranks)):
        idx = pred_ranks[i]
        if idx not in truth and pred[idx] >= pred[ent2idx[q_t]]:
            if (pred[idx] == pred[ent2idx[q_t]]) and (np.random.uniform() < 0.5):
                filtered_ranks.append(idx)
            else:
                filtered_ranks.append(idx)

    rank = len(filtered_ranks) + 1
    return rank


