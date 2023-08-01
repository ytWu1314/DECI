from collections import defaultdict
import os
from typing import DefaultDict
import tqdm
import logging
import sys
from datetime import datetime
import time
import random
import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
import numpy
from argparse import ArgumentParser
import torch.nn.functional as F
import copy

# configuration
parser = ArgumentParser()
# model hyper-parameters
parser.add_argument('--model_name', default='bert-base-uncased', type=str,
                    choices=['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased',
                             'roberta-base', 'roberta-large'])
parser.add_argument('--log', default='training-log.txt', type=str)
parser.add_argument('--max_epoch', default=50, type=int)
parser.add_argument('--gcn_dim', default=200, type=int)
parser.add_argument('--mlp_dim', default=200, type=int)
config = parser.parse_args()

config.command = 'python ' + ' '.join([x for x in sys.argv])

seed = 1987

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=config.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message, printout=True):
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


printlog('Passed args:')
printlog('log path: {}'.format(config.log))
printlog('transformer model: {}'.format(config.model_name))

device = torch.device('cuda')


printlog('Loading graphdata1.pickle')
data = torch.load('graphdata1.pickle')


# DEFINE MODEL

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
embedding_size = 768 if 'base' in config.model_name else 1024
printlog('Defining model')


class GCN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = nn.Linear(input_size, output_size)

    def forward(self, input, graph):
        graph_norm = graph / torch.sum(graph, dim=2)[:, :, None]
        return graph_norm @ self.w(input)


class CausalRelationModel(torch.nn.Module):

    def __init__(self, lstm_size=embedding_size // 2, gcn1_size=config.gcn_dim, gcn2_size=config.gcn_dim,
                 mlp_size=config.mlp_dim):
        super().__init__()

        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.bert_model = AutoModel.from_pretrained(config.model_name).to(device)

        self.gcn1 = GCN(2 * lstm_size, gcn1_size)
        self.gcn2 = GCN(gcn1_size, gcn2_size)

        self.combine_weight = nn.Linear(6 * lstm_size, 7)

        self.transformer_dropout = nn.Dropout(0.4)
        self.gcn_dropout = nn.Dropout(0.5)

        self.mlp0 = nn.Sequential(nn.Linear(2 * gcn2_size, mlp_size),
                                  nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(mlp_size, 2)).to(device)

        self.mlp1 = nn.Sequential(nn.Linear(2 * gcn2_size, mlp_size),
                                  nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(mlp_size, 2)).to(device)

        self.doc_att = nn.Linear(2 * lstm_size, 1)

    def forward(self, document, sentence_mask, event, graph, pair):

        n = len(document)


        ind = torch.LongTensor([0, 2, 3, 4, 5]).to(device)
        graph[ind] = graph[ind] / (torch.sum(graph[ind], dim=[1, 2], keepdim=True) / graph.shape[1] + 1e-5)

        doc_embed = self.bert_embedding(document, sentence_mask)
        doc_embed = self.transformer_dropout(doc_embed)

        ref_graph = graph[1][n:, :n]
        ref_graph_norm = ref_graph / torch.sum(ref_graph, dim=1).view(-1, 1)

        extend_embed = torch.mm(ref_graph_norm, doc_embed)

        embed = torch.cat((doc_embed, extend_embed), dim=0)

        event_e = self.extract_event(embed, event, n)
        doc_e = torch.softmax(self.doc_att(doc_embed), dim=0).T @ doc_embed
        g_e = []

        for i in range(len(pair)):
            u, v = pair[i]
            g_e.append(torch.cat((doc_e.squeeze(), event_e[u], event_e[v])))

        g_e = torch.stack(g_e)

        combine_wei = self.combine_weight(g_e)

        graph1 = torch.softmax(combine_wei, dim=1) @ graph.permute(1, 0, 2)

        graph1 = graph1.permute(1, 0, 2)  # dense graph 66 x 274 x 274

        n_graph = graph1.shape[1]  # 274

        graph2 = torch.eye(n_graph).repeat(graph1.shape[0], 1, 1).to(device)

        graph2[graph1 > 0.1] = 1  # sparse graph 66 x 274 x 274

        dhidden = self.gcn1(embed, graph1)

        dgcn_tune = self.gcn2(dhidden, graph1)
        dgcn_tune = self.gcn_dropout(F.relu(dgcn_tune))

        shidden = self.gcn1(embed, graph2)

        sgcn_tune = self.gcn2(shidden, graph2)
        sgcn_tune = self.gcn_dropout(F.relu(sgcn_tune))

        dp = torch.max(dgcn_tune, dim=1).values
        ds = torch.max(sgcn_tune, dim=1).values

        conloss = torch.mean((dp - ds) ** 2)
        gw = torch.softmax(combine_wei, dim=1)

        event_representation = self.extract_event(dgcn_tune, event, n)

        event_pair_embed0 = []
        event_pair_embed1 = []
        represen = []

        for i in range(len(pair)):

            u, v = pair[i]

            e1 = event_representation[i][u]
            e2 = event_representation[i][v]
            representation = torch.cat((e1 + e2, torch.abs(e1 - e2)))

            represen.append(representation)

            if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:
                event_pair_embed0.append(representation)
            else:
                event_pair_embed1.append(representation)

        if len(event_pair_embed1) == 0:
            event_pair_embed0 = torch.stack(event_pair_embed0)
            prediction0 = self.mlp0(event_pair_embed0)
            return prediction0, conloss, represen, gw

        if len(event_pair_embed0) == 0:

            event_pair_embed1 = torch.stack(event_pair_embed1)
            prediction1 = self.mlp1(event_pair_embed1)
            return prediction1, conloss, represen, gw

        else:

            event_pair_embed0 = torch.stack(event_pair_embed0)
            event_pair_embed1 = torch.stack(event_pair_embed1)

            prediction0 = self.mlp0(event_pair_embed0)
            prediction1 = self.mlp1(event_pair_embed1)

            prediction = []

            i0 = 0
            i1 = 0

            for i in range(len(pair)):
                u, v = pair[i]
                if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:
                    prediction.append(prediction0[i0])
                    i0 += 1
                else:
                    prediction.append(prediction1[i1])
                    i1 += 1

            return torch.stack(prediction), conloss, represen, gw

    def bert_embedding(self, document, sentence_mask):

        document_split = [[] for i in range(sentence_mask[-1] + 1)]
        for i in range(len(document)):
            document_split[sentence_mask[i]].append(document[i])

        n = len(document_split)
        me = 0
        offsets = []

        ids = []

        for sent in document_split:
            id = [101] if config.model_name.startswith('bert') else [0]
            offset = []
            for token in sent:
                token_encode = self.bert_tokenizer.encode(token, add_special_tokens=False)
                offset.append(len(id))
                id += token_encode
            id += [102] if config.model_name.startswith('bert') else [2]

            me = max(me, len(id))
            offsets.append(offset)
            ids.append(id)

        for id in ids:
            id += [1 for i in range(len(id), me)]
        ids = torch.tensor(ids).to(device)
        emb_raw = self.bert_model(ids)[0]

        p = 0
        emb = torch.zeros(len(document), embedding_size).to(device)
        for i in range(len(offsets)):
            offset = offsets[i]
            for j in offset:
                emb[p] = emb_raw[i][j]
                p += 1
        return emb

    def extract_event(self, embed, event, n):

        if len(embed.shape) == 2:
            event_embed = [None for _ in range(len(event))]
            ie = 0
            for i in range(len(event)):
                e = event[i]
                if e != None and len(e) > 0:
                    event_embed[i] = embed[n + ie]
                    ie += 1
            return event_embed

        elif len(embed.shape) == 3:
            event_embed_ = []
            for k in range(embed.shape[0]):
                event_embed = [None for _ in range(len(event))]
                ie = 0
                for i in range(len(event)):
                    e = event[i]
                    if e != None and len(e) > 0:
                        event_embed[i] = embed[k][n + ie]
                        ie += 1
                event_embed_.append(event_embed)
            return event_embed_


# PROCESSING DATA TO NEW FORMAT

train_data_ = data['1'] + data['3'] + data['4'] + data['5'] + data['7'] + data['8'] + data['12'] + data['13'] + data['14'] + data['16'] + data['18'] + data['19'] + data['20'] + data['22'] + data['23'] + data['24'] + data['30'] + data['32'] + data['33'] + data['35']
eval_data_ = data['37'] + data['41']

train_data = []
eval_data = []

batch_size = 1


for idx, (document, sentence_mask, event, pos, neg, graph, eve_coref) in enumerate(train_data_):
    pair = pos + neg   
    label = [1 for i in range(len(pos))] + [0 for i in range(len(neg))]
    xy = [(p, l) for (p, l) in zip(pair, label)]
    random.shuffle(xy)
    
    while len(xy) > 0:
        batch = xy[:batch_size]
        xy = xy[batch_size:]
        pair = [p for (p, l) in batch]
        label = [l for (p, l) in batch]
        
        train_data.append((idx, document, sentence_mask, event, graph, pair, label, eve_coref))

for idx, (document, sentence_mask, event, pos, neg, graph, eve_coref) in enumerate(eval_data_):
    #print(eve_coref)
    pair = pos + neg   
    label = [1 for i in range(len(pos))] + [0 for i in range(len(neg))]
    xy = [(p, l) for (p, l) in zip(pair, label)]
    random.shuffle(xy)
    
    while len(xy) > 0:
        batch = xy[:batch_size]
        xy = xy[batch_size:]
        pair = [p for (p, l) in batch]
        label = [l for (p, l) in batch]

        eval_data.append((idx, document, sentence_mask, event, graph, pair, label, eve_coref))

cr = CausalRelationModel().to(device)

npos0 = 0
nneg0 = 0
npos1 = 0
nneg1 = 0

for idx, document, sentence_mask, event, graph, pair, label, eve_coref in eval_data:

    for i in range(len(pair)):
        u, v = pair[i]

        if label[i] == 1:
            if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:
                npos0 += 1
            else:
                npos1 += 1

        else:
            if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:
                nneg0 += 1
            else:
                nneg1 += 1

w0 = (nneg0 / npos0)
w1 = (nneg1 / npos1) ** 0.75

print(w0, w1)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np

transformer_learning_rate = 2e-5
learning_rate = 1e-4
cross_entropy = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW([
    {'params': cr.bert_model.parameters(), 'lr': transformer_learning_rate},
    {'params': cr.mlp0.parameters(), 'lr': learning_rate},
    {'params': cr.mlp1.parameters(), 'lr': learning_rate},
    {'params': cr.gcn1.parameters(), 'lr': learning_rate},
    {'params': cr.gcn2.parameters(), 'lr': learning_rate},
    {'params': cr.combine_weight.parameters(), 'lr': learning_rate}
])
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=len(train_data) * 5,
                                            num_training_steps=len(train_data) * 50)

printlog('Start training ...')

best_intra = {'p': 0, 'r': 0, 'f1': 0}
best_cross = {'p': 0, 'r': 0, 'f1': 0}
best_intra_cross = {'p': 0, 'r': 0, 'f1': 0}

best_epoch = 0

for epoch in range(config.max_epoch):
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    printlog('Command: {}'.format(config.command))
    torch.cuda.empty_cache()

    t1 = time.time()

    all_label_ = []
    all_predt_ = []
    all_clabel_ = []

    random.shuffle(train_data)

    # TRAIN MODEL
    cr.train()
    progress = tqdm.tqdm(total=len(train_data), ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data)
    step = 0
    optimizer.zero_grad()
    for di in range(len(train_data)):
        progress.update(1)
        idx, document, sentence_mask, event, graph, pair, label, eve_coref = train_data[di]

        if len(pair) < 6:
            continue

        # COMPUTE CLASS WEIGHT
        weight = []
        clabel = []
        for i, (u, v) in enumerate(pair):
            if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:  # intra sentence
                clabel.append(0)
                if label[i] == 0:
                    weight.append(1)
                else:
                    weight.append(w0)

            else:  # cross sentence
                clabel.append(1)
                if label[i] == 0:
                    weight.append(1)
                else:
                    weight.append(w1)

        # RUN MODEL
        prediction, conloss, _, _ = cr(document, sentence_mask, event, graph, pair)

        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()
        all_label_ += label
        all_predt_ += predt
        all_clabel_ += clabel

        label = torch.LongTensor(label).to(device)
        weight = torch.tensor(weight).to(device)
        loss = torch.mean(cross_entropy(prediction, label) * weight + 0.1 * conloss)

        loss.backward()

        step += 1

        printlog('{}/{}: Loss: {}'.format(step, total_step, loss.item()), False)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    progress.close()

    cr.eval()
    # EVALUATE AFTER EACH EPOCH

    all_label = []
    all_predt = []
    all_clabel = []
    # ADD 
    predicts = defaultdict(list)
    golds = defaultdict(dict)
    corefs = {}

    progress = tqdm.tqdm(total=len(eval_data), ncols=75,
                         desc='Eval {}'.format(epoch))
    # CHANGE
    for di in range(len(eval_data)):
        progress.update(1)
        idx, document, sentence_mask, event, graph, pair, label, eve_coref = eval_data[di]

        corefs[idx] = eve_coref

        # COMPUTE INTRA OR CROSS SENTENCE LABEL
        clabel = []
        for i, (u, v) in enumerate(pair):
            if sentence_mask[event[u][0]] == sentence_mask[event[v][0]]:
                clabel.append(0)
                golds[idx][(u, v)] = (0, label[i])
            else:
                clabel.append(1)
                golds[idx][(u, v)] = (1, label[i])

        # RUN MODEL
        prediction, conloss, _, _ = cr(document, sentence_mask, event, graph, pair)

        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()
        for i, (u, v) in enumerate(pair):
            predicts[idx].append(((u, v),predt[i]))

    for id in predicts.keys():
        doc_predicts = predicts[id]
        coref = corefs[id]
        clus_link = []
        for ((u, v), label) in doc_predicts:
            if label == 1:
                for i in range(len(coref)):
                    for j in range(i+1, len(coref)):
                        if u in coref[i] and v in coref[j]:
                            if (set(coref[i]), set(coref[j])) not in clus_link:
                                clus_link.append((set(coref[i]), set(coref[j])))
        
        for ((u, v), label) in doc_predicts:
            if label == 0 and golds[id][(u, v)][0]==1:
                for clus1, clus2 in clus_link:
                    if u in clus1 and v in clus2:
                        label = 1
            
            all_predt.append(label)
            all_label.append(golds[id][(u, v)][1])
            all_clabel.append(golds[id][(u, v)][0])

    progress.close()
    
    exact = [0 for i in range(len(all_label))]
    for i in range(len(all_label)):
        if all_label[i] == 1 and all_label[i] == all_predt[i]:
            exact[i] = 1

    tpi = 0
    li = 0
    pi = 0
    tpc = 0
    lc = 0
    pc = 0

    for i in range(len(exact)):

        if exact[i] == 1:
            if all_clabel[i] == 0:
                tpi += 1
            else:
                tpc += 1

        if all_label[i] == 1:
            if all_clabel[i] == 0:
                li += 1
            else:
                lc += 1

        if all_predt[i] == 1:
            if all_clabel[i] == 0:
                pi += 1
            else:
                pc += 1

    tp = sum(exact)
    l = sum(all_label)
    p = sum(all_predt)

    printlog('-------------------')
    printlog("TIME: {}".format(time.time() - t1))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog("\tprecision score: {}".format(precision_score(all_label_, all_predt_, average=None)[1]))
    printlog("\trecall score: {}".format(recall_score(all_label_, all_predt_, average=None)[1]))
    printlog("\tf1 score: {}".format(f1_score(all_label_, all_predt_, average=None)[1]))

    printlog("EVAL:")

    # INTRA SENTENCE

    printlog('\tINTRA-SENTENCE:')
    recli = tpi / li
    preci = tpi / (pi + 1e-9)
    f1cri = 2 * preci * recli / (preci + recli + 1e-9)

    intra = {
        'p': preci,
        'r': recli,
        'f1': f1cri
    }
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
    printlog("\t\tprecision score: {}".format(intra['p']))
    printlog("\t\trecall score: {}".format(intra['r']))
    printlog("\t\tf1 score: {}".format(intra['f1']))

    # CROSS SENTENCE
    reclc = tpc / lc
    precc = tpc / (pc + 1e-9)
    f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
    cross = {
        'p': precc,
        'r': reclc,
        'f1': f1crc
    }

    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
    printlog("\t\tprecision score: {}".format(cross['p']))
    printlog("\t\trecall score: {}".format(cross['r']))
    printlog("\t\tf1 score: {}".format(cross['f1']))

    # INTRA + CROSS SENTENCE
    intra_cross = {
        'p': precision_score(all_label, all_predt, average=None)[1],
        'r': recall_score(all_label, all_predt, average=None)[1],
        'f1': f1_score(all_label, all_predt, average=None)[1]
    }
    printlog('\tINTRA + CROSS:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi + tpc, pi + pc, li + lc))
    printlog("\t\tprecision score: {}".format(intra_cross['p']))
    printlog("\t\trecall score: {}".format(intra_cross['r']))
    printlog("\t\tf1 score: {}".format(intra_cross['f1']))

    if intra_cross['f1'] > best_intra_cross['f1']:
        printlog('New best epoch...')
        best_intra_cross = intra_cross
        best_intra = intra
        best_cross = cross

        best_epoch = epoch

    printlog('=' * 20)
    printlog('Best result at epoch: {}'.format(best_epoch))
    printlog('Eval intra: {}'.format(best_intra))
    printlog('Eval cross: {}'.format(best_cross))
    printlog('Eval intra cross: {}'.format(best_intra_cross))
