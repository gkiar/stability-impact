#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
import os.path as op
import pandas as pd
import numpy as np
import re

import warnings
warnings.filterwarnings('ignore',
                        category=pd.io.pytables.PerformanceWarning)


def filelist2df(file_list, csv, r):
    def str2list(x): return [float(v) for v in eval(x)]
    def mean(x, t): return np.mean(x).astype(t)

    df = pd.read_csv(csv)
    list_of_dicts = []
    for fl in file_list:

        sim, sub, ses, dirs, pipe, rs = r.match(fl).groups()

        if not len(dirs):
            dirs = "all"
        else:
            dirs = "odd" if dirs == "o" else "even"

        tmp_dict = {}
        tmp_dict["subject"] = sub
        tmp_dict["session"] = ses
        tmp_dict["directions"] = dirs
        tmp_dict["pipeline"] = pipe
        tmp_dict["seed"] = int(rs)
        tmp_dict["simulation"] = sim

        tmp = df.query("subject == '{0}'".format(sub)).iloc[0]
        tmp_dict["age"] = mean(str2list(tmp["calculated_age"]), float)
        tmp_dict["sex"] = mean(str2list(tmp["sex"]), int)
        tmp_dict["bmi"] = mean(str2list(tmp["bmi"]), float)
        tmp_dict["pulse"] = mean(str2list(tmp["pulse"]), int)

        tmp_dict['graph'] = np.loadtxt(fl)
        list_of_dicts.append(tmp_dict)
        del tmp_dict

    ldf = pd.DataFrame(list_of_dicts)
    return ldf


def main(args=None):
    parser = ArgumentParser(__file__,
                            description="Re-formats JSON and matrix data from"
                                        "one-voxel + connectome generation for"
                                        "subsequent analysis.")
    parser.add_argument("graph_dir",
                        help="Corresponding directory containing graphs with "
                             "or without noise injected and stored in the .mat"
                             " ASCII-encoded format. The directory structure "
                             "expected is: "
                             "graph_dir/sim-#/sub-*/ses-*/dwi/*dkt.mat")
    parser.add_argument("participants_csv",
                        help="Table containing list of subjects and their "
                             "phenotypic data.")
    parser.add_argument("output_path",
                        help="Path to the dataframes containing groomed data.")
    parser.add_argument("--ds", action="store_true",
                        help="Flag indiciating whether or not it is a dataset "
                             "which has undergone odd/even downsampling.")

    results = parser.parse_args() if args is None else parser.parse_args(args)

    if results.ds:
        r = re.compile('^.+/(.+)/sub-(.+)/ses-(.+)/dwi/.*dwi_([eo])_(det|prob)_rs([0-9]+)_dkt.mat$')
    else:
        r = re.compile('^.+/(.+)/sub-(.+)/ses-(.+)/dwi/.*dwi_()(det|prob)_rs([0-9]+)_dkt.mat$')

    # Grab and process the graph data
    gd = results.graph_dir
    csv = results.participants_csv

    mat_files = glob(op.join(gd, '*', '*', '*', 'dwi', '*dkt.mat'))
    df_graphs = filelist2df(mat_files, csv, r)

    print(len(df_graphs))
    df_graphs.to_hdf(results.output_path, "graphs", mode="a")


if __name__ == "__main__":
    main()
