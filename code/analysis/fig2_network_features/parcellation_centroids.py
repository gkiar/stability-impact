#!/usr/bin/env python

from argparse import ArgumentParser
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist
import nibabel as nib
import numpy as np
import os.path as op


def gen_centroids(parcellation):
    template = nib.load(parcellation)
    bn = op.splitext(op.splitext(parcellation)[0])[0]

    im = template.get_data()
    rois = np.sort(np.unique(im))
    rois = rois[rois != 0]

    com = []
    for roi in rois:
        com += [center_of_mass((im > 0).astype(int), im, roi)]
    com = np.asarray(com)
    np.savetxt(bn + '_centroids.mat', com)

    dist = cdist(com, com, 'euclidean')
    np.savetxt(bn + '_distances.mat', dist)

    im2 = np.zeros_like(im)
    for _, (x, y, z) in enumerate(com.astype(int)):
        slices = (slice(x-2, x+2), slice(y-2, y+2), slice(z-2, z+2))
        im2[slices] = _ + 1

    centroids = nib.Nifti1Image(im2, template.affine, template.header)
    nib.save(centroids, bn + "_centroids.nii.gz")


def main():
    parser = ArgumentParser()
    parser.add_argument("parcellation", action="store",
                        help="Niftii file of the parcellation to generate "
                             "centroids for. The output will be in the same "
                             "location with the _centroids.mat, "
                             "_centroids.nii.gz, and _distances.mat suffixes.")
    results = parser.parse_args()
    gen_centroids(results.parcellation)


if __name__ == "__main__":
    main()
