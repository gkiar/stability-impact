#!/usr/bin/env python

from argparse import ArgumentParser
from networkx import algorithms as nxa
from networkx import function as nxf
import netneurotools.modularity as nntm
import networkx as nx
import pandas as pd
import os.path as op
import numpy as np


def connection_length(g, dists=None):
    assert(dists is not None)
    adj = nx.adj_matrix(g).toarray()
    dist = dists[adj > 0]
    return dist[dist > 0]


def compute_stats(df_fname, dist_mat=None):
    df = pd.read_hdf(df_fname)
    # Stats organized as follows:
    #   name : lambda/function to be called on (g)
    dists = np.loadtxt(dist_mat)

    stats = {"edgecount": nxf.number_of_edges,
             "globaleffic": nxa.global_efficiency,
             "degree": lambda g: dict(nxf.degree(g)).values(),
             "modularity": lambda g:
                 nntm.consensus_modularity(nx.adj_matrix(g).toarray())[1].mean(),
             "assort": nxa.assortativity.degree_assortativity_coefficient,
             "avplength": lambda g: np.mean(connection_length(g, dists=dists)),
             "weight": lambda g: list(nxf.get_edge_attributes(g, 'weight').values()),
             "ccoeff": lambda g: list(nxa.clustering(g, weight='weight').values()),
             "betweenness": lambda g:
                 list(nxa.betweenness_centrality(g, weight='weight').values()),
             "plength": lambda g: connection_length(g, dists=dists)}

    # create dict (and eventual DF) column per stat with the following struct:
    stat_results = {'index': []}
    stat_results.update({stat_name: [] for stat_name in stats.keys()})

    for idx, row in df.iterrows():
        tmpg = nx.Graph(row.graph)
        stat_results['index'] += [row['index']]
        for stat_name, stat_fn in stats.items():
            tmps = stat_fn(tmpg)
            try:
                len(tmps)
                stat_results[stat_name] += [np.array(list(tmps))]
            except TypeError:
                stat_results[stat_name] += [tmps]

    stat_df = pd.DataFrame.from_dict(stat_results)
    stat_df.to_hdf(df_fname.replace('.h5', '_stats.h5'), "stats", mode="w")


def split_dfs(df_fname, n_rows=10):
    df = pd.read_hdf(df_fname)
    df = df.reset_index(level=0)
    out_fname = df_fname.replace('.h5', '_cut-{0}.h5')
    counter = 0
    start, end = 0, n_rows
    _term = len(df)
    while start <= _term:
        tmpdf = df.iloc[start:end]
        tmpdf.to_hdf(out_fname.format(counter), "graphs", mode='w')
        start += n_rows
        end += n_rows
        counter += 1


def concat_dfs(dfs):
    df = []
    for i, df_fname in enumerate(dfs):
        tmp_df = pd.read_hdf(df_fname)
        df += [tmp_df]
        del tmp_df
    df = pd.concat(df, ignore_index=True)

    fname = op.commonprefix(dfs)
    fname += "_stats.h5"
    df.to_hdf(fname, "stats", mode='w')


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", action="store", type=str,
                        choices=["split", "process", "concat"],
                        help="Split: breaks a dataframe into chunks (requires "
                             "the '--n_rows' argument). Process: computes "
                             "statistics on all graphs in the input dataframe."
                             " Concat: Stacks the graph dataframes together.")
    parser.add_argument("graph_df", action="store", nargs="+", type=str,
                        help="May be one or several (e.g. for 'concat') data"
                             "frames containing graphs to process.")
    parser.add_argument("--n_rows", "-n", action="store", type=int, default=10,
                        help="Number of rows to put in a dataframe when "
                             "splitting.")
    parser.add_argument("--dist_mat", "-d", action="store", type=str,
                        help="Path to .mat file containing Euclidean distances"
                             " between each pair of regions in the atlas.")
    results = parser.parse_args()
    mode = results.mode
    df = results.graph_df
    n_rows = results.n_rows
    d_mat = results.dist_mat

    if mode != "concat":
        df = df[0]

    # dists = np.loadtxt('/data/template/DK_res-1x1x1_distances.mat')
    proc_dict = {"split": lambda _df: split_dfs(_df, n_rows=n_rows),
                 "process": lambda _df: compute_stats(_df, d_mat),
                 "concat": concat_dfs}

    proc_dict[mode](df)


if __name__ == "__main__":
    main()
