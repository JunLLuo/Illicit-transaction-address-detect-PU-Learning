# Data generator

import copy
import pickle
import csv
import json
import networkx as nx
import random

from random import randrange


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class Eth_graph_dataset:
    def __init__(self, file_name):
        self.file_name = file_name
        self.more_labels = None

    def load(self):
        self.G = load_pickle(self.file_name)
        print(nx.info(self.G))

    def _load_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def load_json_auxilary_label(self):
        with open('addresses-darklist.json', 'r') as file:
            more_labels = json.load(file)
            self.more_labels = more_labels

    def sample_nodes_edges(self, n_nodes, nodes_file=None, edges_file=None):
        if not nodes_file:
            nodes_file = f'nodes_eth_{n_nodes}.csv'
        if not edges_file:
            edges_file = f'edges_eth_{n_nodes}.csv'

        header = ['node', 'address', 'isp', 'is_anchor']
        G = self.G
        data = G.nodes
        is_anchors = [0] * (2973489 - 1165) + [1] * 1165
        random.shuffle(is_anchors)
        print(f'length: {len(data)}')

        with open(nodes_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)
            nds = set()
            # add all label 1 from original dataset
            for idx, nd in enumerate(data):
                if data[nd]['isp'] == 1:
                    nds.add(nd)
                    writer.writerow([nd, nd, data[nd]['isp'], is_anchors[idx]])
            print(f'{len(nds)} with label 1 added')

            # add all label 2 from auxilary dataset
            # you can provide more labels for the phishing nodes (with label 2)
            # in addition to the phishing nodes from the dataset (with label 1)
            # otherwise will be 0 nodes with label 2

            nds_second_label = set()
            if self.more_labels is not None:
                more_labels_address = set([d_['address'] for d_ in self.more_labels])
                for idx, nd in enumerate(data):
                    if (data[nd]['isp'] == 0) and (nd in more_labels_address):
                        nds_second_label.add(nd)
                        writer.writerow([nd, nd, 2, is_anchors[idx]])
            print(f'{len(nds_second_label)} with label 2 added')

            have_wrote = 0
            number_limit = copy.deepcopy(n_nodes)

            data_list = list(data)
            for idx, _ in enumerate(data):
                random_index = randrange(len(data))
                nd = data_list[random_index]
                have_wrote = have_wrote + 1
                if have_wrote > number_limit:
                    break
                nds.add(nd)
                writer.writerow([nd, nd, data[nd]['isp'], is_anchors[idx]])
            print(f'{len(nds)} nodes added in total')

            neighbor_nds = set()
            header = ['source', 'target', 'amount', 'timestamp']
            data = nx.edges(G)
            eds = set()
            with open(edges_file, 'w', encoding='UTF8', newline='') as f, open(
                    f"{edges_file.split('.')[0]}_onlyEdges.csv", 'w', encoding='UTF8', newline='') as f_only_edges:
                writer = csv.writer(f)
                writer_only_edges = csv.writer(f_only_edges)
                writer.writerow(header)

                # G[u][v][0] is the gets the first edge from node u to node v.
                # G[u][v][0]['amount'] is the transfered amount,  G[u][v][0]['timestamp'] is the transfered timestep.

                # getting the edges of the selected nodes (like 1165 negative + 1165 positive)
                for Gind, u in enumerate(G):
                    for Vind, v in enumerate(G[u]):
                        if u in nds:
                            neighbor_nds.add(v)
                        elif v in nds:
                            neighbor_nds.add(u)
                        else:
                            continue
                        for Egind, eg_i in enumerate(G[u][v]):
                            eg = G[u][v][eg_i]
                            row = [u, v, eg['amount'], eg['timestamp']]
                            row_only_edges = [u, v]

                            if str(row) not in eds:
                                eds.add(str(row))
                                writer.writerow(row)
                                writer_only_edges.writerow(row_only_edges)

                                # getting the edges of the neighbors of the selected nodes
                for Gind, u in enumerate(G):
                    for Vind, v in enumerate(G[u]):
                        if (u in neighbor_nds) and (v in neighbor_nds):  #
                            for Egind, eg_i in enumerate(G[u][v]):
                                eg = G[u][v][eg_i]
                                row = [u, v, eg['amount'], eg['timestamp']]
                                row_only_edges = [u, v]
                                if str(row) not in eds:
                                    eds.add(str(row))
                                    writer.writerow(row)
                                    writer_only_edges.writerow(row_only_edges)

            print(f'{len(eds)} edges added')


if __name__ == '__main__':
    # This will create three csv files:
    # edges_eth_1165.csv
    # nodes_eth_1165.csv
    # edges_eth_1165_onlyEdges.csv

    ethds = Eth_graph_dataset('./MulDiGraph.pkl')
    ethds.load()
    ethds.sample_nodes_edges(1165, edges_file=None)