# Experiments and scope

## Residual networks

The code implements the architecture and loads official ImageNet weights to classify an example image. The final ResNet-50 print labels are reversed between the custom and torchvision versions; this is a display-label issue in the original solution, not evidence of a new training experiment.

## Transfer learning

The subset contains up to 256 examples per class. The solution unfreezes the classifier and selected late blocks, so it performs **partial fine-tuning**, rather than training only the classifier. No final accuracy log is included in the supplied material.

## ViT attention

The saved figures illustrate attention visualisation. They are not segmentation ground truth or a guarantee of causal explanation. The script replaces an internal `timm` forward method, which can depend on the installed version, and contains a machine-specific `TCL_LIBRARY` setting. Both should be checked when reproducing the experiment on another computer.

## Validation status

The source files have been checked for Python syntax. This does not establish runtime compatibility or reproduce training results. The original mathematical implementation has been preserved.
