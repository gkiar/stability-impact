#!/usr/bin/env python

from argparse import ArgumentParser

from dipy.tracking.streamline import Streamlines
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import ProbabilisticDirectionGetter, peaks_from_model
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, load_trk
from dipy.io import read_bvals_bvecs
from dipy.data import default_sphere
from dipy.tracking import utils
from dipy.viz import has_fury
from nibabel.streamlines import ArraySequence
from onevox.cli import driver as ov


import nibabel as nib
import numpy as np
import os.path as op
import os
import json
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt


def make_descriptor(parser, arguments=None):
    import boutiques.creator as bc

    basename = "dipy_tracking"
    desc = bc.CreateDescriptor(parser, execname=op.basename(basename),
                               tags={"domain": ["neuroinformatics",
                                                "image processing",
                                                "mri", "noise"]})
    desc.save(basename + ".json")

    if arguments is not None:
        invo = desc.createInvocation(arguments)
        invo.pop("boutiques")

        with open(basename + "_inputs.json", "w") as fhandle:
            fhandle.write(json.dumps(invo, indent=4))


def wrap_fuzzy_failures(fn, args=[], kwargs={}, errortype=Exception,
                        failure_threshold=9, verbose=False):
    failure_count = 0
    while True:
        try:
            result = fn(*args, **kwargs)
            break
        except errortype:
            failure_count += 1
            if verbose:
                print("Failure in {0} ({1} of {2})".format(fn.__name__,
                                                           failure_count,
                                                           failure_threshold))
            if failure_count > failure_threshold:
                raise(FloatingPointError("Too many failures; stopping."))
    return result


def tracking(image, bvecs, bvals, wm, seeds, fibers, prune_length=3, rseed=42,
             plot=False, proba=False, verbose=False):
    # Pipelines transcribed from:
    #   https://dipy.org/documentation/1.1.1./examples_built/tracking_introduction_eudx/#example-tracking-introduction-eudx
    #   https://dipy.org/documentation/1.1.1./examples_built/tracking_probabilistic/

    # Load Images
    dwi_loaded = nib.load(image)
    dwi_data = dwi_loaded.get_fdata()

    wm_loaded = nib.load(wm)
    wm_data = wm_loaded.get_fdata()

    seeds_loaded = nib.load(seeds)
    seeds_data = seeds_loaded.get_fdata()
    seeds = utils.seeds_from_mask(seeds_data, dwi_loaded.affine, density=2)

    # Load B-values & B-vectors
    # NB. Use aligned b-vecs if providing eddy-aligned data
    bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    gtab = gradient_table(bvals, bvecs)
    csa_model = CsaOdfModel(gtab, sh_order=6)

    # Set stopping criterion
    gfa = csa_model.fit(dwi_data, mask=wm_data).gfa
    stop_criterion = ThresholdStoppingCriterion(gfa, .25)

    if proba:
        # Establish ODF model
        response, ratio = auto_response(gtab, dwi_data, roi_radius=10,
                                        fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
        csd_fit = csd_model.fit(dwi_data, mask=wm_data)

        # Create Probabilisitic direction getter
        fod = csd_fit.odf(default_sphere)
        pmf = fod.clip(min=0)
        prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                        sphere=default_sphere)
        # Use the probabilisitic direction getter as the dg
        dg = prob_dg

    else:
        # Establish ODF model
        csa_peaks = peaks_from_model(csa_model, dwi_data, default_sphere,
                                     relative_peak_threshold=0.8,
                                     min_separation_angle=45,
                                     mask=wm_data)

        # Use the CSA peaks as the dg
        dg = csa_peaks

    # Create generator and perform tracing
    s_generator = LocalTracking(dg, stop_criterion, seeds, dwi_loaded.affine,
                                0.5, random_seed=rseed)
    streamlines = Streamlines(s_generator)

    # Prune streamlines
    streamlines = ArraySequence([strline
                                 for strline in streamlines
                                 if len(strline) > prune_length])
    sft = StatefulTractogram(streamlines, dwi_loaded, Space.RASMM)

    # Save streamlines
    save_trk(sft, fibers + ".trk")

    # Visualize fibers
    if plot and has_fury:
        from dipy.viz import window, actor, colormap as cmap

        # Create the 3D display.
        r = window.Renderer()
        r.add(actor.line(streamlines, cmap.line_colors(streamlines)))
        window.record(r,
                      out_path=fibers + '.png',
                      size=(800, 800))


def streamlines2graph(streamlines, affine, parcellation, output_file):
    # Load Images
    parcellation_loaded = nib.load(parcellation)
    parcellation_data = parcellation_loaded.get_fdata()

    uniq = np.unique(parcellation_data)
    parcellation_data = parcellation_data.astype(int)
    if list(uniq) != list(np.unique(parcellation_data)):
        raise TypeError("Parcellation labels should be integers.")

    # Perform tracing
    graph, mapping = utils.connectivity_matrix(streamlines, affine,
                                               parcellation_data,
                                               symmetric=True,
                                               return_mapping=True)
    # Deleting edges with the background
    graph = np.delete(graph, (0), axis=0)
    graph = np.delete(graph, (0), axis=1)
    map_keys = sorted(mapping.keys())

    np.savetxt(output_file + ".mat", graph)
    with open(output_file + "_mapping.json", "w") as fhandle:
        for k in map_keys:
            # ignore background fibers
            if 0 in k:
                continue
            v = mapping[k]
            fhandle.write("{0}\t{1}\t{2}\n".format(k[0], k[1],
                                                   ",".join([str(_)
                                                             for _ in v])))

    plt.imshow(np.log1p(graph), interpolation='nearest')
    try:
        plt.savefig(output_file + ".png")
    except ValueError:
        pass


def main(args=None):
    parser = ArgumentParser("dipy_tracking.py",
                            description="Generates streamlines and optionally "
                                        "a connectome from a set of diffusion "
                                        "volumes and parameter files.")
    parser.add_argument("diffusion_image",
                        help="Image containing a stack of DWI volumes, ideally"
                             " preprocessed, to be used for tracing. If this "
                             "is a nifti image, the image is used directly. If"
                             " it is a JSON file, it is expected to be an "
                             "output from the 'oneVoxel' noise-simulation tool"
                             " and the image will be regenerated using the "
                             "parameters contained in the JSON file.")
    parser.add_argument("bvecs",
                        help="The b-vectors corresponding to the diffusion "
                             "images. If the images have been preprocessed "
                             "then the rotated b-vectors should be used.")
    parser.add_argument("bvals",
                        help="The b-values corresponding to the diffusion "
                             "images. ")
    parser.add_argument("whitematter_mask",
                        help="A white matter mask generated from a structural "
                             "image that has been transformed into the same "
                             "space as the diffusion images.")
    parser.add_argument("seed_mask",
                        help="A seed mask, recommended as the white matter and"
                             " gray matter boundary. This can be derived from "
                             "the white matter mask by dilating the image and "
                             "subtracting the original mask.")
    parser.add_argument("output_directory",
                        help="The directory in which the streamlines and "
                             "optionally graphs and figures will be saved in.")
    parser.add_argument("--labels", "-l", nargs="+",
                        help="Optional nifti image containing co-registered "
                             "region labels pertaining to a parcellation. This"
                             " file will be used for generating a connectome "
                             "from the streamlines.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Toggles verbose or quiet output printing.")
    parser.add_argument("--prob", "-P", action="store_true",
                        help="Toggles probabilistic tracking. Default: det.")
    parser.add_argument("--prune", "-p", action="store", type=int, default=3,
                        help="Dictates the minimum length of fibers to keep. "
                             "If fibers are shorter than the value, exclusive,"
                             "then they will be thrown out. Default value is "
                             "3 nodes in the fiber.")
    parser.add_argument("--random_seed", "-r", action="store", type=int,
                        default=42,
                        help="Random seed to be used in tractography.")
    parser.add_argument("--streamline_plot", "-s", action="store_true",
                        help="Toggles the plotting of streamlines. This "
                             "requires VTK.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Toggles creation of a Boutiques descriptor and "
                             "invocation from the tool and inputs.")

    results = parser.parse_args() if args is None else parser.parse_args(args)

    # Just create the descriptor and exit if we set this flag.
    if results.boutiques:
        make_descriptor(parser, results)
        return 0

    verbose = results.verbose
    image = results.diffusion_image
    bn = op.basename(image).split('.')[0]
    outdir = op.join(results.output_directory,
                     bn.split("_")[0],
                     bn.split("_")[1],
                     "dwi")
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    noised = True if image.endswith(".json") else False
    if noised:
        noise_file = image
        # Load noise parameters
        with open(image, 'r') as fhandle:
            noise_data = json.loads(fhandle.read())

        # Apply noise to image
        in_image = noise_data["base_image"]
        ov(in_image, outdir, apply_noise=noise_file, verbose=results.verbose)
        image = noise_file.replace('.json', '.nii.gz')

    rs = results.random_seed
    trackmod = "prob" if results.prob else "det"
    fibers = op.join(outdir, "{0}_{1}_rs{2}".format(bn, trackmod, rs))
    if not op.isfile(fibers + ".trk"):
        wrap_fuzzy_failures(tracking,
                            args=[image, results.bvecs, results.bvals,
                                  results.whitematter_mask, results.seed_mask,
                                  fibers],
                            kwargs={"plot": results.streamline_plot,
                                    "verbose": verbose,
                                    "rseed": rs,
                                    "proba": results.prob},
                            errortype=Exception,
                            failure_threshold=5,
                            verbose=verbose)

    tractog = load_trk(fibers + ".trk", 'same', Space.RASMM)
    streamlines = tractog.streamlines

    if results.labels:
        graphs = []
        for label in results.labels:
            lbn = op.basename(label).split('.')[0].split("_")[-1]
            graphs += [op.join(outdir, "{0}_{1}_rs{2}_{3}".format(bn, trackmod,
                                                                  rs, lbn))]
            streamlines2graph(streamlines, tractog.affine, label, graphs[-1])

    if noised:
        # Delete noisy image
        ov(image, outdir, clean=True, apply_noise=noise_file, verbose=verbose)

    if verbose:
        print("Streamlines: {0}".format(fibers))
        print("Graphs: {0}".format(", ".join(graphs)))


if __name__ == "__main__":
    main()
