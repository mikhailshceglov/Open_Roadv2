"""The stages every method shares.

None of them import a method: they take a ``MethodSpec`` (or, past ``infer``,
only the score maps it wrote), which is what keeps them method-agnostic.

    infer     frames         -> score_raw/*.npy + soft_mask/*.png
    render    score_raw      -> render/{mask,overlay,regions.json}
    evaluate  score_raw + GT -> metrics.json
    report    several runs   -> a comparison table
"""
