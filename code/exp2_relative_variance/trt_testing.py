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

    try:
        assert(np.equal(ys[0], ys[1]).all())
    except AssertionError as e:
        # If they aren't sorted the same, at least make sure they have the same
        # number of members in each group, etc.
        assert(len(ys[0]) == len(ys[1]))
        assert(sum(ys[0]) == sum(ys[1]))

        # If class sizes and things are conserved, just sort them both
        for _ind in [0, 1]:
            tmpy = ys[_ind]
            tmpx = xs[_ind]
            reorder = sorted(range(len(tmpy)), key=lambda k: tmpy[k])

            old_s = xs[_ind].shape
            xs[_ind] = np.array([tmpx[i, :] for i in reorder])
            ys[_ind] = np.array([tmpy[i] for i in reorder])
            assert(old_s == xs[_ind].shape)

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


def hyp1(df_, instrum, ssets, iters, workers):
    stat_list = []
    idvar = "subject"
    ## Hypothesis 1: cross-subject
    hyp = 1

    ### Tests 1 & 2
    for dir_ in df_['directions'].unique():
        ### Test 1: Multi-session, 1 dir, all sims
        test=1
        df__ = df_[df_['directions'] == dir_]
        stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp, test=test,
                                       instrum=instrum, reps=iters,
                                       workers=workers)

        ### Test 2: Multi-session, 1 dir, reference execution
        test=2
        df__ = df__[df__['simulation'] == 'ref']
        stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp, test=test,
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
        stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp, test=test,
                                       instrum=instrum, reps=iters,
                                       workers=workers)

        ### Test 4: Multi-direction, 1 session, reference execution
        test=4
        df_1 = df__[df__['simulation'] == 'ref']
        stat_list += pipeline_discrims(df_1, idvar=idvar, hyp=hyp, test=test,
                                       instrum=instrum, reps=iters,
                                       workers=workers)

        ### Test 5: Multi-simulation, 1 dir, 1 session
        test=5
        for dir_ in df__['directions'].unique():
            df_2 = df__[df__['directions'] == dir_]
            stat_list += pipeline_discrims(df_2, idvar=idvar, hyp=hyp,
                                           test=test, instrum=instrum,
                                           reps=iters, workers=workers)

        del df__, df_1, df_2

    return stat_list


def hyp2(df, instrum, ssets, iters, workers):
    stat_list = []
    idvar = "session"
    ## Hypothesis 2: cross-session
    hyp = 2

    for sub_ in df['subject'].unique():
        df_ = df[df['subject'] == sub_]

        ### Test 6: Multi-direction, all sims
        test = 6
        stat_list += pipeline_discrims(df_, idvar=idvar, hyp=hyp, test=test,
                                       instrum=instrum, reps=iters,
                                       workers=workers)

        ### Test 7: Multi-direction, reference execution
        test = 7
        df__ = df_[df_['simulation'] == 'ref']
        stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp, test=test,
                                       instrum=instrum, reps=iters,
                                       workers=workers)

        ### Test 8: Multi-simulation, 1 dir
        test = 8
        for dir_ in df_['directions'].unique():
            df__ = df_[df_['directions'] == dir_]
            stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp,
                                           test=test, instrum=instrum,
                                           reps=iters, workers=workers)

            del df__

        del df_

    return stat_list


def hyp3(df, instrum, ssets, iters, workers):
    stat_list = []
    idvar = "directions"
    ## Hypothesis 3: cross-direction
    hyp = 3

    for sub_ in df['subject'].unique():
        df_ = df[df['subject'] == sub_]

        for ses_ in df_['session'].unique():
            df__ = df_[df_['session'] == ses_]

            ### Test 9: All sims
            test = 9
            stat_list += pipeline_discrims(df__, idvar=idvar, hyp=hyp,
                                           test=test, instrum=instrum,
                                           reps=iters, workers=workers)

            ### Test 10: 1 sim?

            del df__
        del df_

    return stat_list


def driver(df_fs, df_pyonly, hyp=-1, workers=4, iters=100, seed=42):
    stat_list = []

    for df_ in [df_fs, df_pyonly]:
        instrum = df_['Instrumentation'].values[0]
        ssets = [[(s, df_[df_['subject']==s]['session'].unique()[i])
                  for s in df_['subject'].unique()]
                 for i in range(2)]

        if hyp == -1 or hyp == 1:
            np.random.seed(seed)
            stat_list += hyp1(df_, instrum, ssets, iters, workers)

        if hyp == -1 or hyp == 2:
            np.random.seed(seed)
            stat_list += hyp2(df_, instrum, ssets, iters, workers)

        if hyp == -1 or hyp == 3:
            np.random.seed(seed)
            stat_list += hyp3(df_, instrum, ssets, iters, workers)

    return stat_list


def main():
    parser = ArgumentParser()
    parser.add_argument("h5_perturbed_pipeline")
    parser.add_argument("h5_perturbed_inputs")
    parser.add_argument("csv_discrim_output")
    parser.add_argument("--workers", "-n", default=4)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--iterations", "-r", type=int, default=100)
    parser.add_argument("--hypothesis", "-e", type=int, default=-1,
                        choices=[-1,1,2,3])
    # Hypotheses:
    #  1. Cross-subject variation
    #  2. Cross-session variation
    #  3. Cross-subsample variation
    # -1: All

    results = parser.parse_args()

    df_pyonly = pd.read_hdf(results.h5_perturbed_inputs)
    df_pyonly['Instrumentation'] = "Inputs"

    df_fs = pd.read_hdf(results.h5_perturbed_pipeline)
    df_fs['Instrumentation'] = "Pipeline"

    stat_list = driver(df_fs, df_pyonly, hyp=results.hypothesis,
                       workers=results.workers, seed=results.seed,
                       iters=results.iterations)

    df_stat = pd.DataFrame.from_dict(stat_list)
    df_stat.to_csv(results.csv_discrim_output, index=False)


if __name__ == "__main__":
    main()

