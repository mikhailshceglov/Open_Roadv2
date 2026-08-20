"""RAAS + Objectomaly + semantic fusion: attenuate by class, restore the exceptions.

    method.py       the MethodSpec the skeleton loads
    fusion.py       class attenuation, region measurement, protection -- pure numpy
    clip_regions.py batched CLIP over SAM regions, normal prompts against OOD ones
    refine.py       the bridge to $OBJECTOMALY_ROOT: SAM -> OASC -> MBP
    soft_mask.py    the RAAS elimination formula -- pure numpy
    road_aware.py   road-hole candidate geometry -- pure numpy
    clip_filter.py  the CLIP decision rules for the _id and _ood variants
    backbone.py     the adapter to $RAAS_ROOT (detectron2 + patched Mask2Former)

Everything except method.py, fusion.py and clip_regions.py is shared with the
`raas` and `raas_objectomaly` methods, duplicated rather than imported: the
branches are alternatives that never coexist, and a shared module would have to
live in main, which owns no method code.
"""
