#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
import nibabel as nib
import os.path as op
import numpy as np
import os


def main():
    parser = ArgumentParser()
    parser.add_argument("input_directory")
    results = parser.parse_args()

    indir = results.input_directory

    dirs = glob(op.join(indir, "sub-*", "ses-*", "dwi"))
    file_bundles = [[op.join(d, f) for f in ff if "dwi." in f] for d in dirs for _, __, ff in os.walk(d)]
    for fl in file_bundles:
        fl = sorted(fl)

        # Get volume indices from b0 array
        with open(fl[0]) as fhandle:
            bvals = fhandle.read()
            bvals = np.array([int(b) for b in bvals.split()])
            b0_loc = np.where(bvals == np.min(bvals))[0]
            dwi_loc = np.where(bvals != np.min(bvals))[0]

            eve_loc = np.append(b0_loc, dwi_loc[0::2])
            odd_loc = np.append(b0_loc, dwi_loc[1::2])
            eve_loc.sort(), odd_loc.sort()

        # Using volume indices, split up...
        # bvals...
        eve_bval = " ".join(str(v) for v in bvals[eve_loc].tolist()) + "\n"
        odd_bval = " ".join(str(v) for v in bvals[odd_loc].tolist()) + "\n"
        with open(fl[0].replace(".bvals", "_e.bvals"), 'w') as fhandle:
            fhandle.write(eve_bval)
        with open(fl[0].replace(".bvals", "_o.bvals"), 'w') as fhandle:
            fhandle.write(odd_bval)

        # bvecs...
        with open(fl[1]) as fhandle:
            bvecs = fhandle.readlines()

        eve_bvec = [[val for idx, val in enumerate(line.split()) if idx in eve_loc] for line in bvecs]
        odd_bvec = [[val for idx, val in enumerate(line.split()) if idx in odd_loc] for line in bvecs]
        with open(fl[1].replace(".bvecs", "_e.bvecs"), 'w') as fhandle:
            for line in eve_bvec: fhandle.write(" ".join(str(v) for v in line) + "\n")
        with open(fl[1].replace(".bvecs", "_o.bvecs"), 'w') as fhandle:
            for line in odd_bvec: fhandle.write(" ".join(str(v) for v in line) + "\n")

        # nifti...
        im = nib.load(fl[2])
        eve_im = nib.Nifti1Image(im.get_fdata()[:,:,:,eve_loc], im.affine, im.header)
        odd_im = nib.Nifti1Image(im.get_fdata()[:,:,:,odd_loc], im.affine, im.header)
        nib.save(eve_im, fl[2].replace(".nii.gz", "_e.nii.gz"))
        nib.save(odd_im, fl[2].replace(".nii.gz", "_o.nii.gz"))


if __name__ == "__main__":
    main()

