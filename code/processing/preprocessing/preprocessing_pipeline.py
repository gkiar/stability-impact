#!/usr/bin/env python

from argparse import ArgumentParser
from subprocess import Popen, PIPE
import os
import os.path as op

from bids.layout import BIDSLayout
import nibabel as nib
import numpy as np

import fsl


def execute(cmd, verbose=True, skipif=False):
    if skipif:
        return "Skipping..."
    try:
        print(cmd, flush=True)
        p = Popen(cmd, shell=True, stderr=PIPE, stdout=PIPE)
        log = []
        while True:
            line = p.stdout.readline().decode('utf-8').strip('\n')
            if not line:
                break
            log += [line]
            if verbose:
                print(line, flush=True)
        sout, serr = [tmp.decode('utf-8') for tmp in p.communicate()]
        if serr is not '':
            raise Exception(serr)
    except Exception as e:
        raise(e)
        # Leaving as a blanket raise for now so I can add specific
        # exceptions as they pop up...
    else:
        return log


def makeParser():
    parser = ArgumentParser(__file__, description="Preprocessing pipeline for "
                            "DWI data using FSL's eddy.")
    parser.add_argument("bids_dir", action="store",
                        help="Directory to a BIDS-organized dataset.")
    parser.add_argument("output_dir", action="store",
                        help="Directory to store the preprocessed derivatives.")
    parser.add_argument("analysis_level", action="store", choices=["session"],
                        help="Level of analysis to perform. Options: session")
    parser.add_argument("--participant_label", "-p", action="store", nargs="*",
                        help="Label of the participant(s) to process, omitting "
                        "the 'sub-' portion of the directory name. Supplying "
                        "none means the entire dataset will be processed.")
    parser.add_argument("--session_label", "-s", action="store", nargs="*",
                        help="Label of the session(s) to process, omitting the "
                        "'ses-' portion of the directory name. Supplying none "
                        "means the entire dataset will be processed.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Flag toggling verbose output statements.")
    parser.add_argument("--boutiques", action="store_true",
                        help="Flag toggling descriptor creation.")
    parser.add_argument("--gpu", action="store_true",
                        help="Toggles using GPU accelerated eddy.")
    parser.add_argument("--fsldir", action="store",
                        default="/usr/share/fsl/",
                        help="Path to local installation of FSL. Defaults to "
                             "/usr/share/fsl/.")
    parser.add_argument("--parcellation", "-l", action="store", nargs="+",
                        default=[],
                        help="Parcellation/Label volumes which will be "
                             "transformed into the subject/session DWI space.")
    return parser


def createDescriptor(parser, arguments):
    import boutiques.creator as bc
    import os.path as op
    import json

    desc = bc.CreateDescriptor(parser, execname=op.basename(__file__))
    basename = op.splitext(__file__)[0]
    desc.save(basename + ".json")
    invo = desc.createInvocation(arguments)
    invo.pop("boutiques")

    with open(basename + "_inputs.json", "w") as fhandle:
        fhandle.write(json.dumps(invo, indent=4))


def main():
    parser = makeParser()
    results = parser.parse_args()

    if results.boutiques:
        createDescriptor(parser, results)
        return 0

    verb = results.verbose
    fsldir = results.fsldir
    mni152 = op.join(fsldir, "data", "standard",
                     "MNI152_T1_2mm_brain.nii.gz")
    mni152bn = op.basename(mni152).split(".")[0]

    outdir = results.output_dir
    partis = results.participant_label
    labels = results.parcellation

    if verb:
        print("BIDS Dir: {0}".format(results.bids_dir), flush=True)
        print("Output Dir: {0}".format(results.output_dir), flush=True)
        print("Analysis level: {0}".format(results.analysis_level), flush=True)

    # This preprocessing workflow is modified from the FSL recommendations here:
    #    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide

    # Step 0, 1: Begin interrogation of BIDS dataset

    # Due to current super-linear slowdown in BIDS Layout, exclude all but
    # participant of interest. Explored in the following Github issue:
    #   https://github.com/bids-standard/pybids/issues/285
    if partis is not None:
        pattrn = 'sub-(?!{0})(.*)$'.format("|".join(partis))
    else:
        pattrn = ''

    dset = BIDSLayout(results.bids_dir, exclude=pattrn)
    subjects = dset.get_subjects()
    if results.participant_label is not None:
        subjects = [pl
                    for pl in results.participant_label
                    if pl in subjects]
        assert(len(subjects) > 0)
    if verb:
        print("Participants: {0}".format(", ".join(s for s in subjects)),
              flush=True)

    sessions = dset.get_sessions()
    if results.session_label is not None:
        sessions = [sl
                    for sl in results.session_label
                    if sl in sessions]
        assert(len(sessions) > 0)
    if verb:
        print("Sessions: {0}".format(", ".join(s for s in sessions)), flush=True)

    # Step 0, 2: Prune dataset to subjects/sessions that have necessary files
    ftypes = [".nii", ".bval", ".bvec"]
    collections = []
    for subj in subjects:
        for sess in sessions:
            tf_dwi = dset.get(subject=subj, session=sess,
                              datatype="dwi", suffix="dwi",
                              return_type="file")
            tf_anat = dset.get(subject=subj, session=sess,
                               datatype="anat", suffix="T1w",
                               return_type="file")
            if (all(any(ftype in fl for fl in tf_dwi) for ftype in ftypes) and
                    any(ftypes[0] in fl for fl in tf_anat)):

                    collections += [{"subject": subj,
                                     "session": sess,
                                     "anat": [t
                                              for t in tf_anat
                                              if ftypes[0] in t][0],
                                     "bval": [t
                                              for t in tf_dwi
                                              if ftypes[1] in t][0],
                                     "bvec": [t
                                              for t in tf_dwi
                                              if ftypes[2] in t][0],
                                     "dwi": [t
                                             for t in tf_dwi
                                             if ftypes[0] in t][0]}]
            else:
                if verb:
                    print("Skipping sub-{0}".format(subj) +
                          " / ses-{0} due to missing data.".format(sess),
                          flush=True)

    complete_collection = []
    for col in collections:
        dwibn = op.basename(col["dwi"]).split('.')[0]
        anatbn = op.basename(col["anat"]).split('.')[0]
        subses = op.join('sub-{0}'.format(col['subject']),
                         'ses-{0}'.format(col['session']))

        derivdir_d = op.join(outdir, subses, "dwi")
        derivdir_a = op.join(outdir, subses, "anat")
        execute("mkdir -p {0}".format(derivdir_d),
                verbose=verb,
                skipif=op.isdir(derivdir_d))
        execute("mkdir -p {0}".format(derivdir_a),
                verbose=verb,
                skipif=op.isdir(derivdir_a))

        # Step 1: Extract B0 volumes

        # Make even number of spatial voxels? (req'd for eddy for some reason)
        # TODO : above, if actually needed - docs inconsistent

        # Get B0 locations
        with open(col["bval"]) as fhandle:
            bvals = fhandle.read().split(" ")
            bvals = [int(b) for b in bvals if b != '' and b != '\n']
            b0_loc = [i for i, b in enumerate(bvals) if b == np.min(bvals)]

        # Get B0 volumes
        col["b0_scans"] = []
        for idx, b0 in enumerate(b0_loc):
            b0ind = "b0_{0}".format(idx)
            col["b0_scans"] += [op.join(derivdir_d,
                                        dwibn + "_" + b0ind + ".nii.gz")]
            execute(fsl.fslroi(col["dwi"], col["b0_scans"][-1], *[b0, 1]),
                    verbose=verb, skipif=op.isfile(col["b0_scans"][-1]))

        # Merge B0 volumes
        col["b0s"] = op.join(derivdir_d, dwibn + "_b0s.nii.gz")
        execute(fsl.fslmerge(col["b0s"], *col["b0_scans"]), verbose=verb,
                skipif=op.isfile(col["b0s"]))

        # Create acquisition parameters file
        col["acqparams"] = op.join(derivdir_d, dwibn + "_acq.txt")
        acqs = {"i": "1 0 0", "i-": "-1 0 0",
                "j": "0 1 0", "j-": "0 -1 0",
                "k": "0 0 1", "k-": "0 0 -1"}
        with open(col["acqparams"], 'w') as fhandle:
            meta = dset.get_metadata(path=col["dwi"])
            pedir = meta["PhaseEncodingDirection"]
            trout = meta["TotalReadoutTime"]
            line = "{0} {1}".format(acqs[pedir], trout)
            fhandle.write("\n".join([line] * len(b0_loc)))

        # Step 1.5: Run Top-up on Diffusion data
        # TODO: remove; topup only applies with multiple PEs (rare in open data)
        # col["topup"] = op.join(derivdir_d, dwibn + "_topup")
        # col["hifi_b0"] = op.join(derivdir_d, dwibn + "_hifi_b0")
        # execute(fsl.topup(col["b0s"], col["acqparams"],
        #                   col["topup"], col["hifi_b0"]),
        #         verbose=verb)
        # execute(fsl.fslmaths(col["hifi_b0"], "-Tmean", col["hifi_b0"]),
        #         verbose=verb)

        # Step 2: Brain extraction
        # ... Diffusion:
        col["dwi_brain"] = op.join(derivdir_d, dwibn + "_brain.nii.gz")
        col["dwi_mask"] = op.join(derivdir_d, dwibn + "_brain_mask.nii.gz")
        execute(fsl.bet(col["dwi"], col["dwi_brain"], "-F", "-m"), verbose=verb,
                skipif=op.isfile(col["dwi_brain"]))

        # ... Structural:
        col["anat_brain"] =  op.join(derivdir_a, anatbn + "_brain.nii.gz")
        col["anat_mask"] = op.join(derivdir_a, anatbn + "_brain.nii.gz")
        execute(fsl.bet(col["anat"], col["anat_brain"], "-m"), verbose=verb,
                skipif=op.isfile(col["anat_brain"]))

        # Step 3: Produce prelimary DTIfit QC figures
        col["dwi_qc_pre"] = op.join(derivdir_d, dwibn + "_dtifit_pre")
        execute(fsl.dtifit(col["dwi_brain"], col["dwi_qc_pre"], col["dwi_mask"],
                           col["bvec"], col["bval"]),
                verbose=verb,
                skipif=op.isfile(col["dwi_qc_pre"] + "_FA.nii.gz"))

        # Step 4: Perform eddy correction
        # ... Create index
        col["index"] = op.join(derivdir_d, dwibn + "_eddy_index.txt")
        with open(col["index"], 'w') as fhandle:
            fhandle.write(" ".join(["1"] * len(bvals)))

        # ... Run eddy
        col["eddy_dwi"] = op.join(derivdir_d, dwibn + "_eddy")
        if results.gpu:
            eddy_exe = "eddy_cuda8.0"
        else:
            eddy_exe = "eddy_openmp"
        execute(fsl.eddy(col["dwi_brain"], col["dwi_mask"], col["acqparams"],
                         col["index"], col["bvec"], col["bval"],
                         col["eddy_dwi"], exe=eddy_exe), verbose=verb,
                skipif=op.isfile(col["eddy_dwi"] + ".nii.gz"))

        # Step 5: Registration to template
        # ... Compute transforms
        col["t1w2mni"] = op.join(derivdir_a, anatbn + "_to_mni_xfm.mat")
        execute(fsl.flirt(col["anat_brain"], omat=col["t1w2mni"], ref=mni152),
                verbose=verb,
                skipif=op.isfile(col["t1w2mni"]))

        col["dwi2t1w"] = op.join(derivdir_d, dwibn + "_to_t1w_xfm.mat")
        execute(fsl.flirt(col["eddy_dwi"], ref=col["anat_brain"],
                          omat=col["dwi2t1w"]),
                verbose=verb,
                skipif=op.isfile(col["dwi2t1w"]))

        col["dwi2mni"] = op.join(derivdir_d, dwibn + "_to_mni_xfm.mat")
        execute(fsl.convert_xfm(concat=col["t1w2mni"], inp=col["dwi2t1w"],
                                omat=col["dwi2mni"]),
                verbose=verb,
                skipif=op.isfile(col["dwi2mni"]))

        # ... Invert transforms towards diffusion space
        col["mni2dwi"] = op.join(derivdir_d, dwibn + "_from_mni_xfm.mat")
        execute(fsl.convert_xfm(inverse=col["dwi2mni"], omat=col["mni2dwi"]),
                verbose=verb,
                skipif=op.isfile(col["mni2dwi"]))

        col["t1w2dwi"] = op.join(derivdir_a, anatbn + "_dwi_xfm.mat")
        execute(fsl.convert_xfm(inverse=col["dwi2t1w"], omat=col["t1w2dwi"]),
                verbose=verb,
                skipif=op.isfile(col["t1w2dwi"]))

        # Step 6: Apply registrations to anatomical and template images
        col["anat_in_dwi"] = op.join(derivdir_a, anatbn + "_brain_dwi.nii.gz")
        execute(fsl.flirt(col["anat_brain"], applyxfm=True,
                          out=col["anat_in_dwi"], init=col["t1w2dwi"],
                          ref=col["eddy_dwi"]),
                verbose=verb,
                skipif=op.isfile(col["anat_in_dwi"]))

        col["mni_in_dwi"] = op.join(derivdir_d,
                                    ("atlas_" + dwibn + "_" +
                                     mni152bn + "_dwi.nii.gz"))
        execute(fsl.flirt(mni152, applyxfm=True, out=col["mni_in_dwi"],
                          init=col["mni2dwi"], ref=col["eddy_dwi"]),
                verbose=verb,
                skipif=op.isfile(col["mni_in_dwi"]))

        # Step 7: Perform tissue segmentation on anatomical images in DWI space
        col["tissue_masks"] = op.join(derivdir_d, anatbn + "_fast")
        execute(fsl.fast(col["anat_in_dwi"], col["tissue_masks"],
                classes=3, imtype=1),
                verbose=verb,
                skipif=op.isfile(col["tissue_masks"] + "_seg_2.nii.gz"))

        # Step 8: Transform parcellations into DWI space
        col["labels_in_dwi"] = []
        for label in labels:
            lbn = op.basename(label).split('.')[0]
            col["labels_in_dwi"] += [op.join(derivdir_d,
                                             ("labels_" + dwibn + "_" +
                                              lbn + ".nii.gz"))]
            execute(fsl.flirt(label, applyxfm=True,
                              out=col["labels_in_dwi"][-1],
                              init=col["mni2dwi"], ref=col["eddy_dwi"],
                              interp="nearestneighbour"),
                    verbose=verb,
                    skipif=op.isfile(col["labels_in_dwi"][-1]))

        if verb:
            print("Finished processing sub-{0}".format(subj) +
                  " / ses-{0} !".format(sess), flush=True)
        complete_collection += [col]


if __name__ == "__main__":
    main()
