#!/usr/bin/env python

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import os.path as op
import os
from scipy.optimize import minimize
import itertools


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def objective(var, cov, ind):
    #             0            1          2          3            4
    # var = [   x-sub,       x-ses,     x-dir,     x-pipe,      x-mca   ]
    # aka    w/in-homeos,   w/in-sub,  w/in-ses,  w/in-dir,   w/in-pipe

    # This requires that "cov" and "ind" are defined in the global namespace
    model_cov = np.zeros_like(cov)
    model_cov[ind["xsub"]] += var[0]
    model_cov[ind["xses"]] += var[1]
    model_cov[ind["xdir"]] += var[2]
    model_cov[ind["xpipe"]] += var[3]
    if ind.get("xmca"):
        model_cov[ind["xmca"]] += var[4]

    return rmse(model_cov, cov)


def gentle_get_index(df, query, crash=False):
    try:
        indices = df.loc[query]["index"]
        return list(indices)
    except TypeError:
        return [indices]
    except KeyError as e:
        if crash:
            raise e
        else:
            return []


def get_variance_indices(df, mca=False):
    cols = ["xsub", "xses", "xdir", "xpipe", "xmca"]
    indices = {c: np.array([[], []], dtype=int) for c in cols}
    levels = df.index.levels

    def to_idx(li):
        tmp = np.array(list(itertools.permutations(sorted(li), 2)),
                       dtype=int)
        try:
            return tmp[:, 0], tmp[:, 1]
        except IndexError:
            return np.array([li, li], dtype=int)

    # Create lists of rows for all those matching a given criteria
    # cross subject: everything
    indices["xsub"] = to_idx(list(df['index'].values))

    # cross session: all within each subject
    for su in levels[0]:
        indices["xses"] = np.append(indices["xses"],
                                    to_idx(gentle_get_index(df, su)),
                                    axis=1)

        # cross dirs: all within each session
        for se in levels[1]:
            try:
                indices["xdir"] = np.append(indices["xdir"],
                                            to_idx(gentle_get_index(df, (su, se),
                                                                    crash=True)),
                                            axis=1)
            except KeyError:
                # This means that a given session wasn't found for that person
                # in this case, don't bother trying to go deeper
                continue

            # cross pipe: all within each scan/set of dirs
            for di in levels[2]:
                indices["xpipe"] = np.append(indices["xpipe"],
                                             to_idx(gentle_get_index(df, (su, se, di))),
                                             axis=1)

                # cross mca: all within each pipeline
                if not mca:
                    continue

                for pi in levels[3]:
                    indices["xmca"] = np.append(indices["xmca"],
                                                to_idx(gentle_get_index(df, (su, se, di, pi))),
                                                axis=1)
    if not mca:
        del indices["xmca"]
    return indices
