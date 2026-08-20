"""The four distillation stages, in the order ``entrypoint.sh`` runs them.

    label     teacher -> float16 anomaly maps + quarter-res semantic logits
    train     student on those maps
    evaluate  SMIYC metrics + teacher fidelity on unlabelled frames
    export    ONNX + latency

Each is idempotent: it writes a marker on success and is skipped on re-run, so
a job that dies in training does not re-pay for labelling.
"""
