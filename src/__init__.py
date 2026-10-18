"""CodeEmbed: Small Code Embedding Model."""

import os
import sys

# Prevent PyTorch threadpool deadlock under Scalene signal profiling on macOS
if "scalene" in sys.modules or os.environ.get("SCALENE_PROFILING"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        if sys.platform == "darwin":
            torch.set_num_threads(1)
    except ImportError:
        pass

__version__ = "0.1.0"
