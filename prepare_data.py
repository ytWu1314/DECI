import time
import random
import torch
import torch.nn as nn
import pickle
from transformers import BertTokenizer, BertModel
import os

random.seed(1235)

device = torch.device('cuda')


import pickle
with open('data.bin', 'rb') as f:
    data = pickle.load(f)


embedding_size = 300

import fasttext.util
from nltk.corpus import wordnet as wn

ft = fasttext.load_model('cc.en.300.bin')

import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def extract_graph(document, sentence_mask, dep, ent_coref, eve_coref, wsd, event):

    embed = []
    
    for token in document:
        embed.append(ft[token])

    embed = torch.tensor(embed).to(device)

    n_event = sum([span != None and len(span) > 0 for span in event])
    n_entity = sum([len(coref) for coref in ent_coref])
    
    n = len(document) + n_event + n_entity


    dep_graph = torch.zeros(n, n).to(device)
    ref_graph = torch.zeros(n, n).to(device)
    coref_graph = torch.zeros(n, n).to(device)
    wordnet_graph = torch.zeros(n, n).to(device)
    bert_sem_graph = torch.zeros(n, n).to(device)
    event_entity_graph = torch.zeros(n, n).to(device)


    # graph for dependency tree

    pred, dept = dep

    rang = torch.arange(len(dept)).to(device)
    dept = torch.tensor(dept).to(device)


    dep_graph[dept, rang] = 1
    dep_graph[rang, dept] = 1


    # graph for event and entity mention

    ie = len(document)
    event_embed = []

    for span in event:
        if span != None and len(span) > 0:
            ee = torch.zeros(embedding_size).to(device)

            for s in span:
                ee += embed[s]
                ref_graph[ie][s] = 1
                ref_graph[s][ie] = 1

            ie += 1

            ee /= len(span)
            event_embed.append(ee)


    entity_embed = []

    for coref in ent_coref:
        for ref in coref:
            ee = torch.zeros(embedding_size).to(device)

            for s in ref:
                ee += embed[s]
                ref_graph[ie][s] = 1
                ref_graph[s][ie] = 1
            
            ie += 1

            ee /= len(ref)
            entity_embed.append(ee)


    # graph for event coreference

    ofs = len(document)

    for coref in eve_coref:
        nco = len(coref)
        for i in range(nco - 1):
            for j in range(i + 1, nco):
                u = ofs + coref[i]
                v = ofs + coref[j]
                coref_graph[i][j] = 1
                coref_graph[j][i] = 1
    

    
    # graph for entity coreference

    ofs = len(document) + n_event

    for coref in ent_coref:
        nco = len(coref)
        for i in range(ofs, ofs + nco - 1):
            for j in range(i + 1, ofs + nco):
                coref_graph[i][j] = 1
                coref_graph[j][i] = 1
        ofs += nco


    
    # graph for wordnet similarity

    for i in range(len(wsd) - 1):
        for j in range(i + 1, len(wsd)):
            _, synu, u = wsd[i]
            _, synv, v = wsd[j]
            sim = wn.synset(synu.name()).wup_similarity(wn.synset(synv.name()))
            if sim and sim > 0.7:
                wordnet_graph[u][v] = 1
                wordnet_graph[v][u] = 1


    cross = torch.mm(embed, embed.T) / torch.sqrt(torch.sum(embed * embed, dim = 1).view(-1, 1) * torch.sum(embed * embed, dim=1).view(1, -1))
    cross -= torch.eye(embed.shape[0]).to(device)

    bert_sem_graph[:embed.shape[0], :embed.shape[0]][cross >= 1] = 1

    for i in range(len(document) - 1):
        for j in range(i + 1, len(document)):
            if document[i] in stopwords or document[j] in stopwords:
                bert_sem_graph[i][j] = 0
                bert_sem_graph[j][i] = 0

    # graph for entity, event intra sentence

    event_in_sentence = []
    for span in event:
        if span != None and len(span) > 0:
            event_in_sentence.append(sentence_mask[span[0]])

    entity_in_sentence = []
    for coref in ent_coref:
        for ref in coref:
            entity_in_sentence.append(sentence_mask[ref[0]])


    eve_ofs = len(document)
    ent_ofs = len(document) + n_event

    for ev_id, evs_id in enumerate(event_in_sentence):
        for en_id, ens_id in enumerate(entity_in_sentence):
            if evs_id == ens_id :
                event_entity_graph[eve_ofs + ev_id][ent_ofs + en_id] = 1
                event_entity_graph[ent_ofs + en_id][eve_ofs + ev_id] = 1


    # eye graph

    eye_graph = torch.eye(n).to(device)

    return torch.stack([dep_graph, ref_graph, coref_graph, wordnet_graph, bert_sem_graph, event_entity_graph, eye_graph])

idx = 0
import time

for key in data:
    for i in range(len(data[key])):
        t1 = time.time()
        document, sentence_mask, event, positive, negative_sample, dep, ent_coref, eve_coref, wsd = data[key][i]
        graph = extract_graph(document, sentence_mask, dep, ent_coref, eve_coref, wsd, event)
        data[key][i] = [document, sentence_mask, event, positive, negative_sample, graph, eve_coref]

        print('Processing ', idx, time.time() - t1); t1 = time.time(); idx += 1


print('dumping data ...')
torch.save(data, 'graphdata1.pickle')

del ft
import gc
gc.collect()
