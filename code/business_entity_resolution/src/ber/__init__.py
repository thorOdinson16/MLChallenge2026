"""Business Entity Resolution pipeline (Amazon ML Challenge 2026)."""
import os

# transformers tries to import TensorFlow when it is present; we never use it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
