import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from extraction import findSVOs
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
import en_core_web_lg
nlp = spacy.load('en_core_web_lg')

def create_graph(final_nodes):
    source = []
    target = []
    edge = []
    indexes = []

    for i in (range(len(final_nodes))):
        ent1 = final_nodes[i][0]
        ent2 = final_nodes[i][2]
        rel = final_nodes[i][1] 
        source.append(ent1.lower().strip())
        target.append(ent2.lower().strip())
        edge.append("".join(rel).strip())
        indexes.append(i)
    if(len(edge) == 0 or len(final_nodes) == 0):
        return None
    else:
        G = nx.DiGraph(directed=True)
        for i in (range(len(edge))):
            G.add_weighted_edges_from([(source[i], target[i], i)])
        size=20
        if len(edge)/2 > 20:
            size = len(edge)/2
        plt.figure(figsize = (size, size))
        edge_labels = dict([((u, v, ), edge[d['weight']]) for u, v, d in G.edges(data = True)])
        pos = nx.spring_layout(G, k = 0.8)
        nx.draw(G, with_labels = True, node_color = 'lightblue', node_size=5000, edge_color='r', edge_cmap = plt.cm.Blues, pos=pos, font_size=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 15)
        return G, edge_labels

def load_text(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = sent_tokenize(text)
    return sentences
    
# for i in nodes:
#     for j in i:
#         if(len(j) == 3):
#             final_nodes.append(j)

# g = create_graph(final_nodes)
# plt.savefig('graph.png')

# def make_graph_sum(summary):
#     nodes = []
#     final_nodes = []
#     sentences = sent_tokenize(summary) 
#     for sentence in sentences: 
#         tokens = nlp(sentence)
#         svos = findSVOs(tokens)
#         nodes.append(svos)

#     for i in nodes:
#         for j in i:
#             if(len(j) == 3):
#                 final_nodes.append(j)
#     g = create_graph(final_nodes)
#     plt.savefig('static/sgraph.png')

def make_graph(text):
    nodes = []
    final_nodes = []
    sentences = sent_tokenize(text) 
    for sentence in sentences: 
        tokens = nlp(sentence)
        svos = findSVOs(tokens)
        nodes.append(svos)

    for i in nodes:
        for j in i:
            if(len(j) == 3):
                final_nodes.append(j)
    g = create_graph(final_nodes)
    # plt.savefig('static/graph.png')
    new_graph_name = "graph" + str(time.time()) + ".png"
    for filename in os.listdir('static/'):
        if filename.startswith('graph_'):  # not to remove other images
            os.remove('static/' + filename)
    plt.savefig('static/' + new_graph_name)
    return new_graph_name