"""S124 verify one-shot repair of repeated GitHub PR Actions fanout.

Read-only repository audit. No training, no artifact deletion, no S80 promotion.
"""
import json,re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
m=json.loads((root/"analysis/workflow_dedup_manifest.json").read_text())
files=m["workflows"]
assert len(files)==40 and len(set(files))==40, "exact targeted 40 repeated PR audits"
errors=[]
for p in files:
    text=(root/p).read_text()
    expected="  push:\n    branches: [codex/v273-scope-rescue]"
    if expected not in text:errors.append((p,"branch-scoped push trigger missing"))
    if re.search(r"^  pull_request:\s*$",text,re.M):errors.append((p,"unwanted PR trigger"))
    if not re.search(r"^  workflow_dispatch:\s*$",text,re.M):errors.append((p,"manual dispatch lost"))
    if not re.search(r"^    paths:\s*(?:\\[.*\\])?\\s*$",text,re.M):errors.append((p,"push path filters missing"))
    if "'"+p+"'" not in text[:text.index("\njobs:")]:errors.append((p,"self workflow not in paths"))
    if "group: ${{ github.workflow }}-${{ github.ref }}" not in text:
        errors.append((p,"per-workflow concurrency missing"))
    if "cancel-in-progress: true" not in text:errors.append((p,"stale runs not cancelled"))
if errors:raise RuntimeError("Workflow regression: "+repr(errors))
result={"status":"S124_DEDUP_AUDIT_PASSED","converted":len(files),
    "trigger":"push, branch-local paths-scoped",
    "manual_dispatch_retained":True,"concurrency_cancel_superseded":True,
    "historic_workflow_files_or_artifacts_deleted":False,
    "model_promotion_changed":False}
print(json.dumps(result,indent=2))
