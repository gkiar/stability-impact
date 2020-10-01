#!/usr/bin/env python

import re
import pandas as pd

r1 = re.compile('sub-(\w+)_ses-(\w+)_(\w+).nii.gz')
r2 = re.compile('^(DS|CLG|NFB|ALG)(1|2|3|A|2R|AR|4|4R|5)$')

filename = '/data/RocklandSample/assessments/gkanalysis/rockland_log.txt'
with open(filename) as fhandle:
    data = fhandle.readlines()

data_dicts = []
for d in data:
    m = r1.match(d)
    if m:
        g = m.groups()
        study, code = r2.match(g[1]).groups()

        if g[2] == "T1w":
            mod = (1, 0)
        elif g[2] == "dwi":
            mod = (0, 1)
        else:
            mod = (0, 0)

        data_dicts += [{"subject": g[0],
                        "study": study,
                        "code": code,
                        "T1w": mod[0],
                        "DWI": mod[1]}]

# print(data_dicts)
# print(len(data_dicts))

df = pd.DataFrame.from_dict(data_dicts)
df = df.groupby(['subject',
                 'study',
                 'code'])[['T1w', 'DWI']].sum().reset_index()
df.to_csv('/data/RocklandSample/assessments/gkanalysis/scan_visit_list.csv',
          index=False)
