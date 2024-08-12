try:
    from .fno2d import FNO2D
except ImportError:
    print("neural operators not installed. Install it with 'pip install neuraloperator'.")

from .gfno2d import GFNO2d
