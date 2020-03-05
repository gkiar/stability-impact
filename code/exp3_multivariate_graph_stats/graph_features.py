#!/usr/bin/env python

from argparse import ArgumentParser
from network import algorithms as nxa
from network import function as nxf
import networkx as nx
import pandas as pd
import os.path as op
import numpy as np
import os


def get_graphs(graph_dir):
    graph_files = os.listdir(op.abspath(graph_dir))
    graph_list = []
    # For each graph file, open it and store the filename + graph
    for graph_name in graph_files:
        tmp_adj = np.loadtxt(graph_name)
        graph_list = [{"filename": graph_name,
                       "graph": nx.graph.Graph(tmp_adj,
                                               weighted=True,
                                               directed=False)}]
        # Verify that the adjacency matrix created from the NetworkX
        # representation matches the original adjacency matrix
        assert tmp_adj == nx.adj_matrix(graph_list[-1]["graph"]).todense()
    return graph_list


def compute_summaries(graphs):
    # Stats organized as follows:
    #   name : lambda/function to be called on (g)
    stats = {"edgecount": nxf.number_of_edges,
             "globaleffic": nxa.global_efficiency,
             "diameter": nxa.diameter,
             "degree": nxf.degree,
             "assort": nxa.assortativity.degree_assortativity_coefficient,
             "avplength": nxa.average_shortest_path_length,
             "weight": lambda g: nxf.get_edge_attributes(g, 'weight').values(),
             "ccoeff": lambda g: nxa.clustering(g, weight='weight').values(),
             "betweenness": lambda g:
                 nxa.betweenness_centrality(g, weight='weight').values(),
             "plength": lambda g: [_
                                   for t in nxa.shortest_path_length(g)
                                   for _ in t[1].values()
                                   if _ > 0]}

    # create dict (and eventual DF) column per stat with the following struct:
    #  unique ID | feature1 value | feature2 value | feature3_value ....
    feature_dicts = []
    for graph in graphs:
        tmp_dict = {}
        g = graph["graph"]
        f = op.basename(graph["filename"])
        for stat_name in stats.keys():
            tmp_dict["filename"] = f
            tmp_dict["graph"] = g
            tmp_dict[stat_name] = stats[stat_name](g)
            tmp_dict
        feature_dicts += [tmp_dict]

    feature_df = pd.DataFrame.from_dict(feature_dicts)
    return feature_df


def concat_summaries(summary_list):
    dfs = []
    for feature_df in summary_list:
        dfs += [pd.read_hdf(feature_df, 'features')]

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("outfile", action="store", type=str,
                        help="Location of resulting data frame")
    parser.add_argument("--graph_dir", "-g", action="store", nargs=1, type=str,
                        help="Corresponding directory containing graphs with "
                             "or without noise injected and stored in the .mat"
                             " ASCII-encoded format. The directory structure "
                             "expected is: graph_dir/graphs.mat")
    parser.add_argument("--concat", "-c", action="append", nargs=1, type=str,
                        help="")
    results = parser.parse_args()

    if results.concat is not None:
        feature_df = concat_summaries(results.concat)

    elif results.graph_dir is not None:
        graph_list = get_graphs(results.graph_dir)
        feature_df = compute_summaries(graph_list)

    else:
        print("Insufficient arguments provided -- specify graph dir or "
              "summaries to concatenate. Stopping.")
        return -1

    feature_df.to_hdf(results.outfile, "features", mode='w', format="fixed")


if __name__ == "__main__":
    main()
