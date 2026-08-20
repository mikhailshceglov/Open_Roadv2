"""Namespace for the methods on this branch.

Empty on ``main``, and it stays empty: a method branch adds a subdirectory here
and nothing else, which is what lets branches merge without conflicts. This
file exists only so ``methods.<name>.method`` is importable, letting a method's
internals use relative imports (``from .student.model import Student``) instead
of being injected onto ``sys.path`` -- where two methods both shipping a
``student`` package would shadow each other.

See ``open_road.registry`` for how a method is found.
"""
