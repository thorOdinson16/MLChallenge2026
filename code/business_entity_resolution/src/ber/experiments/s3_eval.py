"""Holdout-only stage-3 evaluation with the friend's overnight harness (no test build).
    python -m ber.experiments.s3_eval ce_v4,ce_xlmr_fr[,ce_full]"""
import sys
from . import overnight

if __name__ == "__main__":
    overnight.PARAMS["num_threads"] = 8          # leave CPU for the GPU job's tokeniser
    overnight.choose(sys.argv[1].split(","))
