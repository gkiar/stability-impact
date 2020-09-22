#!/usr/bin/env python

import os
import os.path as op
from glob import glob
import json
from argparse import ArgumentParser
from copy import deepcopy
import random


def get_files(basenames, template, exp):
    invos = []
    for fl in basenames:
        tinv = deepcopy(template)

        tinv["diffusion_image"] = fl
        tinv["bvecs"] = fl.replace(".nii.gz", ".bvecs")
        tinv["bvals"] = fl.replace(".nii.gz", ".bvals")
        lab = [fl.replace("dwi/", "anat/").replace("dwi", "T1w_dkt"),
               fl.replace("dwi/", "anat/").replace("dwi", "T1w_aparc+aseg")]
        if exp.startswith("multi"):
            tinv["labels"] = [l.replace("_o.n", ".n").replace("_e.n", ".n")
                              for l in lab]
        else:
            tinv["labels"] = lab

        tinv["whitematter_mask"] = tinv["labels"][0].replace("dkt", "wm")
        tinv["seed_mask"] = tinv["labels"][0].replace("dkt", "wm_boundary")

        if exp == "multi_pipeline" or exp == "age_bmi":
            tinv["prob"] = [True, False]
            tinv["random_seed"] = 42

        elif exp == "multi_seed":
            tinv["prob"] = True
            tinv["random_seed"] = list(range(42, 52))
        invos += [tinv]
    return invos


def main():
    parser = ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir_base")
    parser.add_argument("template_invocation")
    parser.add_argument("n_sims", type=int)
    parser.add_argument("exp", choices=["multi_pipeline", "multi_seed",
                                        "age_bmi"])
    results = parser.parse_args()

    indir = results.input_dir
    outdir = results.output_dir_base
    n_sims = results.n_sims
    exp = results.exp
    invo = results.template_invocation
    invodir = './invocations-{0}'.format(exp)
    try:
        os.makedirs(invodir)
        os.makedirs(invodir + "-ref")
    except FileExistsError:
        pass

    pttn = op.join(indir, 'sub-*', 'ses-*', 'dwi', '*dwi*.nii.gz')
    basenames = sorted(glob(pttn))

    with open(invo) as fhandle:
        template = json.load(fhandle)

    invos = get_files(basenames, template, exp)

    for idx, invo in enumerate(invos):
        invo["output_directory"] = []
        for jdx in range(n_sims):
            invo["output_directory"] += [op.join(outdir, "sim-{0}".format(jdx))]
        outpath = op.join(invodir, "invo-{0}.json".format(idx))
        with open(outpath, 'w') as fhandle:
            fhandle.write(json.dumps(invo))

        # Add a reference execution
        invo["output_directory"] = op.join(outdir, "ref")
        invo["VFC_BACKENDS"] = "libinterflop_ieee.so"
        outpath = op.join(invodir + "-ref", "invo-{0}_ref.json".format(idx))
        with open(outpath, 'w') as fhandle:
            fhandle.write(json.dumps(invo))

if __name__ == "__main__":
    main()
