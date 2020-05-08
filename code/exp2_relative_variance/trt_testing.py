#!/usr/bin/env python

import pandas as pd
import numpy as np
import os.path as op
import os

from hyppo import discrim
from argparse import ArgumentParser


stat_cols = ['hypothesis', 'test', 'pipeline', 'instrumentation',
             'discrim', 'p-value']


def pipeline_discrims(df_, idvar='subject', hyp=0, test=0, instrum='NA',
                      reps=100, workers=4):
    stat_list = []
    xs = []
    ys = []
    for pipe in ['det', 'prob']:
        df = df_[df_['pipeline'] == pipe]
        x = np.array([np.reshape(g, -1) for g in df.graph.values])
        xs += [x]

        # Determine membership
        uniq = list(df[idvar].unique())
        y = np.array([uniq.index(s) for s in df[idvar]])
        ys += [y]

        disc, pval = discrim.DiscrimOneSample().test(x, y, reps=reps,
                                                     workers=workers)
        tmpstat = {stat_cols[0]: hyp,
                   stat_cols[1]: test,
                   stat_cols[2]: pipe,
                   stat_cols[3]: instrum,
                   stat_cols[4]: disc,
                   stat_cols[5]: pval}

        stat_list += [tmpstat]
        del tmpstat

    assert(np.equal(ys[0], ys[1]).all())
    d1, d2, pval = discrim.DiscrimTwoSample().test(xs[0], xs[1], y, reps=reps,
                                                   workers=workers)
    tmpstat = {stat_cols[0]: hyp,
               stat_cols[1]: test,
               stat_cols[2]: 'det-prob',
               stat_cols[3]: instrum,
               stat_cols[4]: d1-d2,
               stat_cols[5]: pval}
    stat_list += [tmpstat]
    del tmpstat

    return stat_list


def driver(df_fs, df_pyonly, workers=4, iters=100, seed=42):
    stat_list = []
    np.random.seed(seed)

    for df_ in [df_fs, df_pyonly]:
        instrum = df_['Instrumentation'].values[0]
        ssets = [[(s, df_[df_['subject']==s]['session'].unique()[i])
                  for s in df_['subject'].unique()]
                 for i in range(2)]

        ## Hypothesis 1: cross-subject
        hyp=1

        ### Tests 1 & 2
        for dir_ in df_['directions'].unique():
            ### Test 1: Multi-session, 1 dir, all sims
            test=1
            df__ = df_[df_['directions'] == dir_]
            stat_list += pipeline_discrims(df__, hyp=hyp, test=test,
                                           instrum=instrum, reps=iters,
                                           workers=workers)

            ### Test 2: Multi-session, 1 dir, reference execution
            test=2
            df__ = df__[df__['simulation'] == 'ref']
            stat_list += pipeline_discrims(df__, hyp=hyp, test=test,
                                           instrum=instrum, reps=iters,
                                           workers=workers)

            del df__

        ### Tests 3, 4 & 5
        for subses in ssets:
            ### Test 3: Multi-direction, 1 session, all sims
            test=3
            qbase = '(subject == "{0}" and session == "{1}")'
            q = [qbase.format(s[0], s[1]) for s in subses]
            df__ = pd.concat([df_.query(q_) for q_ in q])
            stat_list += pipeline_discrims(df__, hyp=1, test=test,
                                           instrum=instrum, reps=iters,
                                           workers=workers)

            ### Test 4: Multi-direction, 1 session, reference execution
            test=4
            df_1 = df__[df__['simulation'] == 'ref']
            stat_list += pipeline_discrims(df_1, hyp=1, test=test,
                                           instrum=instrum, reps=iters,
                                           workers=workers)

            ### Test 5: Multi-simulation, 1 dir, 1 session
            test=5
            for dir_ in df__['directions'].unique():
                df_2 = df__[df__['directions'] == dir_]
                stat_list += pipeline_discrims(df_2, hyp=1, test=test,
                                               instrum=instrum, reps=iters,
                                               workers=workers)

            del df__, df_1, df_2

    return stat_list


def main():
    parser = ArgumentParser()
    parser.add_argument("h5_perturbed_pipeline")
    parser.add_argument("h5_perturbed_inputs")
    parser.add_argument("csv_discrim_output")
    parser.add_argument("--workers", "-n", default=4)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--iterations", "-r", type=int, default=100)
    results = parser.parse_args()

    df_pyonly = pd.read_hdf(results.h5_perturbed_inputs)
    df_pyonly['Instrumentation'] = "Inputs"

    df_fs = pd.read_hdf(results.h5_perturbed_pipeline)
    df_fs['Instrumentation'] = "Pipeline"

    stat_list = driver(df_fs, df_pyonly, workers=results.workers,
                       seed=results.seed, iters=results.iterations)

    df_stat = pd.DataFrame.from_dict(stat_list)
    df_stat.to_csv(results.csv_discrim_output, index=False)


if __name__ == "__main__":
    main()

