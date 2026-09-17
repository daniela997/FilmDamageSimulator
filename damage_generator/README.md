#Damage generator

The code in this directory allows for the generation of new damage masks of arbitrary size using our statistical model of analogue damage. In addition to the extracted real artifacts, synthetic artifacts can also be used in the generated masks.


## Synthetic artifacts

The optional `--synthetic` flag mixes additional synthetic artifact overlays into the
generated masks. These overlay assets are **not distributed with this repository**.

To use the flag, place your own RGBA PNG overlays in a `synthetic/` directory at the
repository root, with one subdirectory per artifact type:

```
synthetic/
├── dirt/
├── dots/
├── hair/
├── hair-short/
├── lint/
├── scratches/
├── smut/
├── spots/
├── sprinkles/
└── stain/
```

The alpha channel is used as the artifact mask. Without `--synthetic`, masks are
generated from the real annotated artifacts in `scans/` only, and no `synthetic/`
directory is needed.
