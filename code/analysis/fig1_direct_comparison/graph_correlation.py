#!/usr/bin/env python

import numpy as np


def corr(*gs):
    shp = np.product(gf.shape)
    for _, gg in enumerate(gs):
        if _ == 0:
            g = np.reshape(gg, (shp, 1))
        else:
            g = np.append(g, np.reshape(gg, (shp, 1)), axis=1)
    g = g.T
    return np.corrcoef(g)

