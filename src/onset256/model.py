"""Pin verified 256-trained S613 and S701 weights; no frozen S708 1024 head."""
import hashlib
from pathlib import Path
import joblib
from .acoustics import load_native_heads

S613_SHA256="7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f"
S701_SHA256="cc72cf81f477c6b734e253df5fb04d1789ab707dc160f06bdc5cf9ffdff2fb0f"
def _check(path,sha):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=sha:raise ValueError("Untrusted Onset 256 checkpoint: "+Path(path).name)
    return sha
def load(native_path,s701_path):
    _check(native_path,S613_SHA256);_check(s701_path,S701_SHA256)
    native=load_native_heads(native_path,expected_sha256=S613_SHA256)
    if "selection" not in native.metadata:raise ValueError("S613 256 selection missing")
    # Only project-owned, SHA-pinned GitHub Actions #39085408073 joblib.
    specialist=joblib.load(s701_path)
    if getattr(specialist,"n_features_in_",None)!=39:raise ValueError("S701 expected 39D")
    return native,specialist
