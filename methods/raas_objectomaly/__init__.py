"""RAAS with SAM-guided boundary refinement (Objectomaly).

    method.py      the MethodSpec the skeleton loads
    refine.py      the bridge to $OBJECTOMALY_ROOT: SAM -> OASC -> MBP
    soft_mask.py   the RAAS elimination formula -- pure numpy, no models
    road_aware.py  road-hole candidate geometry -- pure numpy
    clip_filter.py the CLIP decision rules for the _id and _ood variants
    backbone.py    the adapter to $RAAS_ROOT (detectron2 + patched Mask2Former)

soft_mask, road_aware, clip_filter and backbone are shared with the `raas`
method, duplicated rather than imported: the branches are alternatives that
never coexist, and a shared module would have to live in main, which owns no
method code.
"""
