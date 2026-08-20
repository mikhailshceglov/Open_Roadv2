"""RAAS-Distill: a real-time student distilled from the RAAS teacher.

    method.py     the MethodSpec the skeleton loads
    student/      the model and its preprocessing -- all that inference needs
    profile.py    end-to-end latency, per step and per module
    train/        the distillation pipeline that produced the weights
    weights/      student_final.pt, 14 MB
"""
