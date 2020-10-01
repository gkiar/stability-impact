#!/usr/bin/env python

from glob import glob
import pandas as pd

cpu_hrs = lambda sec: sec/3600
cpu_yrs = lambda hr: hr/(168*52)

def main():
    files = glob('./*.json')
    total_hrs = 0
    total_yrs = 0
    colname = 'Time: Total (s)'
    for fl in files:
        df = pd.read_json(fl)
        hr = cpu_hrs(df[colname].sum())
        yr = cpu_yrs(hr)

        print("{0}: {1} hrs | {2} yrs".format(fl, hr, yr))

        total_hrs += hr
        total_yrs += yr

    print("Total: {0} hrs | {1} yrs".format(total_hrs, total_yrs))

if __name__ == "__main__":
    main()
