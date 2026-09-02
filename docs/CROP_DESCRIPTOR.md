# 48-D crop descriptor

`dial_det.adapters.crop_descriptor.dial48_crop_descriptor` implements an
explicit 48-D reference descriptor:

- four 6-bin unsigned-gradient cells: 24 dimensions;
- twelve normalized radial Fourier-energy bands: 12 dimensions;
- RGB mean and standard deviation: 6 dimensions;
- luminance contrast, robust Laplacian noise, saturation mean/standard
  deviation, colorfulness, and normalized luminance entropy: 6 dimensions.

The identifier is `dial48-reference-v1` and belongs in feature-tap metadata.
The article states the dimensional partition but does not print every numerical
image-processing definition. Therefore an exact article-level checkpoint must
record the descriptor implementation it used; the repository never assumes
that equal dimensionality alone proves byte-identical features.
