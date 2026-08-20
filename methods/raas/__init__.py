"""RAAS / Maskomaly: anomaly by elimination over Mask2Former queries.

    method.py      the MethodSpec the skeleton loads
    soft_mask.py   the elimination formula -- pure numpy, no models
    road_aware.py  road-hole candidate geometry -- pure numpy
    clip_filter.py the CLIP decision rules for the _id and _ood variants
    backbone.py    the adapter to $RAAS_ROOT (detectron2 + patched Mask2Former)
"""
