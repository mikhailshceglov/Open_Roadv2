"""The distillation pipeline that produced ``weights/student_final.pt``.

Nothing under ``methods/raas_distill`` outside this package imports it, and
inference does not need it. It is kept because the weights are only as
trustworthy as the process that made them, not because it runs out of the box:
the labelling and evaluation stages need the RAAS monorepo, which is not
vendored here. See ``README.md`` in this directory.
"""
