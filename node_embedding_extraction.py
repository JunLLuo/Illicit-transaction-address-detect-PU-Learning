import pandas as pd


def load_poincare_embeddings(filename):
    # If you would like to obtain Poincare embeddings
    # An good implementation is provided at  https://github.com/lateral/geodesic-poincare-embeddings

    fin = open(filename, 'r')
    vectors = {}

    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if vec[1:]:
            vectors[vec[0]] = [float(x) for x in vec[1:]]
        else:
            l_ = fin.readline()
            vec_ = l_.strip().split(' ')
            vectors[vec[0]] = [float(x) for x in vec_[0:]]
    fin.close()

    return vectors


def get_vector(src, addr, node_to_index, vectors=None, X=None):
    if src == 'local':
        return vectors[addr]
    elif src == 'karateclub_model':
        return X[node_to_index[addr]]
    else:
        return None


def get_embeddings(nodes_df, src, node_to_index, samples=1165, X=None):
    embeddings_dict = {}
    embeddings = []
    labels = []

    if samples:
        nodes_df = pd.concat([nodes_df[nodes_df['isp'] == 1], nodes_df[nodes_df['isp'].isin([0])].sample(n=samples)])

    successful_rows = []
    for i, row in nodes_df.iterrows():
        try:
            embeddings_dict[row['node']] = {
                'vector': get_vector(src, row['node'], node_to_index, X=X),
                'label': row['isp']
            }
        except KeyError as e:
            pass
        else:
            embeddings.append(get_vector(src, row['node'], node_to_index, X=X))
            labels.append(row['isp'])
            successful_rows.append(row)
    new_df = pd.DataFrame(successful_rows)

    return embeddings, labels, embeddings_dict, new_df





