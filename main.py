import pandas as pd
import networkx as nx

from tqdm import tqdm
from sqlitedict import SqliteDict
from karateclub import Node2Vec, Role2Vec, MNMF
from node_embedding_extraction import get_embeddings

from utils import *
from classifiers import get_LR_results, get_Elkanoto_PU_results, get_BaggingPu_results


if __name__ == '__main__':
    # run data_processor to get the following csvfiles.
    edges_ = 'edges_eth_1165.csv'
    nodes_ = 'nodes_eth_1165.csv'

    print(f'Loading edge file from {edges_}')
    df_edges_for_indexing = pd.read_csv(edges_)

    node_to_index = {}
    index = 0

    source_k = 'source'
    target_k = 'target'

    df_cached = SqliteDict("local_cache_for_df.sqlite", autocommit=True)
    if 'df_edges_for_indexing_sorted' in df_cached:
        print(f'Loading df from local cache')
        df_edges_for_indexing_sorted = df_cached['df_edges_for_indexing_sorted']
        node_to_index = df_cached['node_to_index']
    else:
        print(f'Making addresses to indices of addresses')
        for i, row in tqdm(df_edges_for_indexing.iterrows()):
            if row[source_k] not in node_to_index:
                node_to_index[row[source_k]] = index
                index = index + 1
            if row[target_k] not in node_to_index:
                node_to_index[row[target_k]] = index
                index = index + 1

        # this is for making the edge df
        # source,target,amount,timestamp
        # from address_a, address_b,1.08896154,1499283988.0
        # to 0, 1, 1.088962, 1.499284e+09

        for i, row in df_edges_for_indexing.iterrows():
            df_edges_for_indexing.at[i, source_k] = node_to_index[row[source_k]]
            df_edges_for_indexing.at[i, target_k] = node_to_index[row[target_k]]

        df_edges_for_indexing_sorted = df_edges_for_indexing.sort_values(by=['timestamp'])
        df_edges_for_indexing_sorted = df_edges_for_indexing_sorted.reset_index(drop=True)
        df_cached['df_edges_for_indexing_sorted'] = df_edges_for_indexing_sorted
        df_cached['node_to_index'] = node_to_index

    nodes_df = pd.read_csv(nodes_)
    nodes_df = nodes_df[['node', 'isp']]

    # load node representation learning model
    g = nx.from_pandas_edgelist(df_edges_for_indexing_sorted, source=source_k, target=target_k, create_using=nx.DiGraph())
    print("Loaded graph for the node representation learning model", nx.info(g))

    # Can use any karateclub supported models such as for this paer: Node2Vec, Role2Vec, MNMF
    model = Role2Vec(walk_number=10, walk_length=5, dimensions=64)
    model.fit(g)
    X = model.get_embedding()

    # sample is how many class 0 to train and plot
    embeddings, labels, embeddings_dict, nodes_df_sampled = get_embeddings(
        nodes_df, src='karateclub_model', node_to_index=node_to_index, samples=None, X=X
    )
    raw_labels = copy.deepcopy(labels)

    print(f'embeddings shape: {np.shape(embeddings)}')

    dict_counter = dict(Counter(nodes_df_sampled['isp']))
    print(f'Node addresses: {dict_counter}')

    results_dict = {'LR': {},'BaggingPu': {}, 'ElkanotoPU': {}}
    for i in range(0, 80, 5):
        X_train, X_test, y_train, y_test, y_test_neg, y_test_pos, labels_ = split_and_label_pu_data(0, i, raw_labels, embeddings)

        results_dict['LR'][i] = get_LR_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos)
        results_dict['BaggingPu'][i] = get_BaggingPu_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos)
        results_dict['ElkanotoPU'][i] = get_Elkanoto_PU_results(X_train, X_test, y_train, y_test, y_test_neg, y_test_pos)

    print(results_dict)
