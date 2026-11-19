import json

import numpy as np
import pytest

from h0_location import write_json
from scripts.decoder_evidence_geometry import error_rows
from scripts.diagnose_decoder_evidence import INT64_FIX_PREVIOUS_SCRIPTS, record_protocol


def test_error_event_bounds_are_serializable_in_case_cache(tmp_path):
    gt = np.zeros((64, 64), dtype=bool)
    gt[20, 4:60] = True
    gt[40, 4:60] = True
    bridged = gt.copy()
    bridged[20:41, 32] = True

    events = error_rows("case", gt, {"on": bridged, "off": gt}, tolerance=4, minimum=5)
    assert len(events) == 1
    assert events[0]["on_category"] == "bridge_candidate"
    assert events[0]["off_category"] == "absent"

    output = tmp_path / "evidence.json"
    write_json(output, {"errors": events})
    assert json.loads(output.read_text(encoding="utf-8"))["errors"] == events


def test_int64_fix_resumes_only_matching_previous_protocol(tmp_path):
    output = tmp_path / "protocol.json"
    current = {"threshold": 0.45, "checkpoint_sha256": "test", "scripts_sha256": {"script": "new"}}
    previous = dict(current, scripts_sha256=INT64_FIX_PREVIOUS_SCRIPTS)
    write_json(output, dict(previous, threshold=0.50))
    with pytest.raises(ValueError, match="different run"):
        record_protocol(output, current)

    write_json(output, previous)
    record_protocol(output, current)
    assert json.loads(output.read_text(encoding="utf-8")) == current
