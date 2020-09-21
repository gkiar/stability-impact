### Deterministic Processing with Dipy

After preprocessing the raw diffusion data using FSL, structural connectomes are generated for the DKT parcellation using Dipy.
The workflow for processing was adopted from the 'Getting Started' guide on the Dipy website, [here](http://nipy.org/dipy/examples_built/introduction_to_basic_tracking.html).

In addition to the eddy-corrected diffusion volumes, rotated b-vectors, and associated b-values from the diffusion sequence, a
white matter and seed mask were needed for tracing. The white matter mask was generated using FSL's fast tissue segmentation tool
on the skull-stripped T1w image linearly-aligned to the subject's diffusion space using three tissue classes. In order to seed from the
gray-white matter boundary, a seed mask was created by taking the difference between a dilated and eroded version of the white
matter mask, resulting in a boundary that is 2 voxels thick.

After loading the necessary data, a six-component CSA-ODF model was fit to the diffusion data residing within white matter. Seeds
were generated in a 2x2x2 arrangement for each voxel within the boundary mask, resulting in 8 seeds per boundary voxel. Deterministic
tracing was then performed using a half-voxel step size, and streamlines shorter than 3-points in length were discarded as spurious.

Once streamlines were generated, they were traced through parcellations translated from MNI space to subject-diffusion space by way of
a linear transformation between the subject-T1w space and MNI space, and between subject-diffusion and subject-T1w space. Edges were
added to the graph corresponding to the end-points of each fiber, and were weighted by the streamline count.
