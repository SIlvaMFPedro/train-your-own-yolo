# ===============================
# MISCELLANEOUS UTILITY FUNCTIONS
# ===============================

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from functools import reduce


# -----------------------------
#   FUNCTIONS
# -----------------------------
def compose(*funcs):
    """
        Compose arbitrarily many functions, evaluated left to right.
    """
    # Return lambda x: reduce(lambda v: f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

