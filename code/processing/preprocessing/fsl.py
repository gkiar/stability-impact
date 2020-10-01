#!/usr/bin/env python

from argparse import ArgumentParser
from subprocess import Popen, PIPE
import os.path as op
import os


def bet(inp, outp, *flags):
    if len(flags):
        flags = " ".join([str(f) for f in flags])
    else:
        flags = ""

    return ("bet "
            "{0} "
            "{1} "
            "{2}".format(inp, outp, flags))


def dtifit(dwi, output, mask, bvec, bval):
    return ("dtifit "
            "-k {0} "
            "-o {1} "
            "-m {2} "
            "-r {3} "
            "-b {4} ".format(dwi, output, mask, bvec, bval))


def eddy(dwi, mask, acq, ind, bvec, bval, out, exe="eddy"):
    return ("{} ".format(exe) +
            "--imain={0} "
            "--mask={1} "
            "--acqp={2} "
            "--index={3} "
            "--bvecs={4} "
            "--bvals={5} "
            "--repol "
            "--out={6}".format(dwi, mask, acq, ind, bvec, bval, out))

def fast(image, output_base, classes=3, imtype=1):
    return ("fast "
            "-t {0} "
            "-n {1} "
            "-o {2} "
            "-g "
            "{3}"
            "".format(imtype, classes, output_base, image))

def flirt(inp, ref="/usr/share/fsl/data/standard/MNI152_T1_2mm_brain",
          bins=256, cost="corratio", searchrx="-90 90", searchry="-90 90",
          searchrz="-90 90", dof=12, applyxfm=False, paddingsize=0.0,
          interp="trilinear", out=None, omat=None, init=None):
    if applyxfm:
        assert(init is not None)
        assert(out is not None)
        return ("flirt "
                "-applyxfm "
                "-in {0} "
                "-init {1} "
                "-out {2} "
                "-ref {3} "
                "-paddingsize {4} "
                "-interp {5} "
                "".format(inp, init, out, ref, paddingsize, interp))
    else:
        assert(omat is not None)
        return ("flirt "
                "-in {0} "
                "-ref {1} "
                "-omat {2} "
                "-bins {3} "
                "-cost {4} "
                "-searchrx {5} "
                "-searchry {6} "
                "-searchrz {7} "
                "-dof {8}"
                "".format(inp, ref, omat, bins, cost, searchrx, searchry,
                          searchrz, dof))


def convert_xfm(inp=None, omat=None, inverse=None, concat=None):
    # convert_xfm(concat=t12mnixfm, inp=dwi2t1xfm, omat=totalxfm)
    # convert_xfm(inverse=totalxfm, inverse=reversexfm)
    if inverse:
        return ("convert_xfm "
                "-inverse {0} "
                "-omat {1}"
                "".format(inverse, omat))
    elif concat:
        return ("convert_xfm "
                "-concat {0} "
                "-omat {1} "
                "{2}"
                "".format(concat, omat, inp))


def fslmaths(*args):
    # This isn't super helpful since fslmaths can do anything...
    return ("fslmaths "
            "{0}".format(" ".join([str(a) for a in args])))


def fslmerge(outp, *inps):
    try:
        assert(len(inps) > 1)
        inps = " ".join(str(i) for i in inps)
        return ("fslmerge "
                "-t {0} "
                "{1}".format(outp, inps))
    except AssertionError as e:
        raise SystemExit("Improper arguments provided. Pls read docs")


def fslroi(inp, outp, *loc):
    try:
        assert(len(loc) == 2 or len(loc) == 6)
        assert(all(type(l) == int for l in loc))
        loc = " ".join(str(l) for l in loc)
        return ("fslroi "
                "{0} "
                "{1} "
                "{2}".format(inp, outp, loc))
    except AssertionError as e:
        raise SystemExit("Improper arguments provided. Pls read docs")


def topup(b0s, acq, outp, bmask, mode="b02b0.cnf"):
    try:
        return ("topup "
                "--imain={0} "
                "--datain={1} "
                "--config={4} "
                "--out={2} "
                "--iout={3}".format(b0s, acq, outp, bmask, mode))
    except AssertionError as e:
        raise SystemExit("Improper arguments provided. Pls read docs")
