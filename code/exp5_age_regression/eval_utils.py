#!/usr/bin/env python


from scipy.spatial.distance import pdist
import numpy as np
import itertools


def prune(df, sample=False, indices=[], verbose=False, **kvpairs):
    # Modes (and combinations thereof):
    #  1) all from a subject
    #  2) all from a pipeline
    #  3) sampled dataset, 1 scan per person (dep: simid or random seed)
    #  4) specific slice

    newdf = df.copy(deep=True)

    # Case 1: we know what we want to slice, just do it
    if indices:
        return newdf.loc[indices]

    # Case 2: randomly sample the dataset
    if sample:
        # N.B.: If true, should be an int
        assert(isinstance(sample, int))

        # save the random state and set a new one
        prev_state = np.random.get_state()
        np.random.seed(sample)

        # get the number of options we're sampling from, and sample
        def fn(x): return x.loc[np.random.choice(x.index, 1, replace=False), :]
        newdf = newdf.groupby(['subject', 'session'], as_index=False).apply(fn)

        # restore the random state
        np.random.set_state(prev_state)

    # Case 3:
    if kvpairs:
        query_terms = []
        for k, v in kvpairs.items():
            if isinstance(v, list):
                eq, v = v[0], v[1]
            else:
                eq, v = "==", v
            query_terms += ["{0} {1} '{2}'".format(k, eq, v)]
        query_str = " and ".join(query_terms)
        if verbose:
            print(query_str)
        newdf = newdf.query(query_str)

    return newdf


def percdev_map(df):
    df = df.copy(deep=True).reset_index()
    # extracts and flattens graphs
    gs = np.array([g for g in df.graph.values])
    gs = np.reshape(gs, (gs.shape[0], gs.shape[1]**2))

    def norm(x): return np.linalg.norm(x)

    # creates container for normalized norms
    fro = {"x_mca": [],
           "x_dir": [],
           "x_ses": [],
           "x_sub": []}

    # filters reference executions only
    df_ref = prune(df, simulation="ref")

    # for all subjects...
    i_t0 = set(df.index)
    to_ignore = set()
    for sub in df["subject"].unique():
        ref = norm(np.mean(prune(df_ref, subject=sub).graph.values))
        df_t1 = prune(df, subject=sub)
        i_t1 = set(df_t1.index)

        # for all sessions...
        for ses in df_t1["session"].unique():
            df_t2 = prune(df_t1, session=ses)
            i_t2 = set(df_t2.index)

            # for all sets of directions...
            for dirs in df_t2["directions"].unique():
                df_t3 = prune(df_t2, directions=dirs)
                i_t3 = set(df_t3.index)

                # X-MCA: pairwise between G in this table
                i_xmca = i_t3
                fro['x_mca'] += (pdist(gs[list(i_xmca)],
                                       metric='euclidean')*100.0/ref).tolist()

                # X-DIR: pairwise between G here and remaining G in df_t2
                i_xdir = i_t2 - i_t3 - to_ignore
                fro['x_dir'] += [norm(gs[p1] - gs[p2])*100.0/ref
                                 for p1, p2 in itertools.product(i_t3, i_xdir)]

                # X-SES: pairwise between G here and uncompared from same sub
                i_xses = i_t1 - i_t2 - to_ignore
                fro['x_ses'] += [norm(gs[p1] - gs[p2])*100.0/ref
                                 for p1, p2 in itertools.product(i_t3, i_xses)]

                # X-SUB: pairwise between G here and all others uncompared
                i_xsub = i_t0 - i_t1 - to_ignore
                fro['x_sub'] += [norm(gs[p1] - gs[p2])*100.0/ref
                                 for p1, p2 in itertools.product(i_t3, i_xsub)]

                # we've exhaustively used i_t3, so now ignore them
                to_ignore = to_ignore | i_t3
    return fro


def corr_map(df):
    df = df.copy(deep=True).reset_index()
    # extracts graphs and computes correlation
    gs = np.array([g for g in df.graph.values])
    cs = corr(gs)

    # creates container for normalized norms
    corrs = {"x_mca": [],
             "x_dir": [],
             "x_ses": [],
             "x_sub": []}

    # for all subjects...
    i_t0 = set(df.index)
    to_ignore = set()
    for sub in df["subject"].unique():
        df_t1 = prune(df, subject=sub)
        i_t1 = set(df_t1.index)

        # for all sessions...
        for ses in df_t1["session"].unique():
            df_t2 = prune(df_t1, session=ses)
            i_t2 = set(df_t2.index)

            # for all sets of directions...
            for dirs in df_t2["directions"].unique():
                df_t3 = prune(df_t2, directions=dirs)
                i_t3 = set(df_t3.index)

                # X-MCA: pairwise between G in this table
                i_xmca = i_t3
                corrs['x_mca'] += [cs[p, q]
                                   for p, q in itertools.product(i_xmca,
                                                                 i_xmca)]

                # X-DIR: pairwise between G here and remaining G in df_t2
                i_xdir = i_t2 - i_t3 - to_ignore
                corrs['x_dir'] += [cs[p, q]
                                   for p, q in itertools.product(i_xmca,
                                                                 i_xdir)]

                # X-SES: pairwise between G here and uncompared from same sub
                i_xses = i_t1 - i_t2 - to_ignore
                corrs['x_ses'] += [cs[p, q]
                                   for p, q in itertools.product(i_xmca,
                                                                 i_xses)]

                # X-SUB: pairwise between G here and all others uncompared
                i_xsub = i_t0 - i_t1 - to_ignore
                corrs['x_sub'] += [cs[p, q]
                                   for p, q in itertools.product(i_xmca,
                                                                 i_xsub)]

                # we've exhaustively used i_t3, so now ignore them
                to_ignore = to_ignore | i_t3
    return corrs


def sigdig_map(df, base=10, masked=False):
    df = df.copy(deep=True).reset_index()
    # extracts and flattens graphs
    gs = np.array([g for g in df.graph.values])

    # creates container for normalized norms
    sigs = {"x_mca": [],
            "x_dir": [],
            "x_ses": [],
            "x_sub": []}

    b = base  # base for sigdig
    m = masked
    # for all subjects...
    i_t0 = set(df.index)

    # X-SUB: pairwise between G here and all others uncompared
    sigs["x_sub"] += [sigdig(gs[list(i_t0)], base=b, masked=m)]
    for sub in df["subject"].unique():
        df_t1 = prune(df, subject=sub)
        i_t1 = set(df_t1.index)

        # X-SES: pairwise between G here and uncompared from same sub
        sigs["x_ses"] += [sigdig(gs[list(i_t1)], base=b, masked=m)]
        # for all sessions...
        for ses in df_t1["session"].unique():
            df_t2 = prune(df_t1, session=ses)
            i_t2 = set(df_t2.index)

            # X-DIR: pairwise between G here and remaining G in df_t2
            sigs["x_dir"] += [sigdig(gs[list(i_t2)], base=b, masked=m)]
            # for all sets of directions...
            for dirs in df_t2["directions"].unique():
                df_t3 = prune(df_t2, directions=dirs)
                i_t3 = set(df_t3.index)

                # X-MCA: pairwise between G in this table
                sigs["x_mca"] += [sigdig(gs[list(i_t3)], base=b, masked=m)]

    return sigs


def sigdig(gs, base=10, masked=False):
    eps = np.finfo(np.float64).eps
    if base == 10:
        log = np.log10
    elif base == 2:
        log = np.log2
    elif base == 'e':
        log = np.log
    else:
        raise ValueError('please pick one of 10, 2, or "e" as your base')

    n = gs.shape[0]

    def c4(n):
        try:
            return np.sqrt(2/(n-1))*np.math.gamma(n/2)/np.math.gamma((n-1)/2)
        except OverflowError:
            # approximation for cases where n is large:
            return 1.0 - (.25/n)

    s = np.std(gs, axis=0)/c4(n) + eps
    mu = np.mean(gs, axis=0) + eps

    sig_gs = -log(s/mu + eps)
    sig_gs = np.nan_to_num(sig_gs)
    sig_gs[sig_gs < 0] = 0

    # the below blacks out th map at mean=0 locations
    if masked:
        mask = mu
        mask[mask > eps] = 1
        mask[mask <= eps] = np.nan
        mask[mask == 1] = 0
        return sig_gs + mask
    else:
        return sig_gs


def nan_weighted_mean(arr, axis=0):
    copy = np.nan_to_num(arr)
    nanmask = np.nanmean(arr, axis=axis)
    nanmask[nanmask >= 0] = 0
    return np.mean(copy, axis=axis) + nanmask


def corr(gs):
    shp = np.product(gs[0, :, :].shape)
    g = np.reshape(gs, (gs.shape[0], shp))
    return np.corrcoef(g)
