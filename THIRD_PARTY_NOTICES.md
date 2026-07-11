# Third-party notices

This repository contains or depends on third-party software and datasets.

## Waymo Open Dataset protocol-buffer modules

The generated Python modules under `src/waymo_open_dataset/` were generated
from protocol definitions distributed by the Waymo Open Dataset project. The
Waymo Open Dataset code repository is licensed under the Apache License 2.0
(except separately identified upstream subdirectories that are not included
here). The raw Waymo Open Dataset is governed by separate dataset terms and is
not redistributed by this repository.

Upstream project: `waymo-research/waymo-open-dataset`.

## CommonRoad software

CommonRoad packages are installation-time dependencies and are not vendored in
this repository. They retain their upstream licenses. In particular,
`commonroad-io` is distributed under the BSD 3-Clause License. Consult each
installed CommonRoad package for its exact license and version.

## Third-party datasets

The Waymo Open Motion Dataset and CommonRoad scenario collections are
third-party datasets. Users must obtain them from the official providers and
comply with the applicable terms. No raw Waymo or CommonRoad data are included
in this repository.
