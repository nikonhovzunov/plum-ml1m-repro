import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from plum_ml1m.decoding import TrieConstrainedDecoder
from plum_ml1m.eval import EvaluationCase, compute_generative_diagnostics_from_cases
from plum_ml1m.eval.diagnostics import (
    DiagnosticSelection,
    GenerativeDiagnosticsError,
    build_generative_diagnostics_report,
)
from plum_ml1m.sid import ItemSIDMapping
from plum_ml1m.trie import TokenTrie

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "reports" / "snapshots"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def _tiny_decoder() -> TrieConstrainedDecoder:
    sids = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=np.int64,
    )
    mapping = ItemSIDMapping.from_sids(sids)
    token_ids_to_sid = {
        (11, 12, 13, 14): (1, 2, 3, 4),
        (21, 22, 23, 24): (5, 6, 7, 8),
    }
    trie = TokenTrie.from_sequences(token_ids_to_sid.keys(), eos_id=99)
    return TrieConstrainedDecoder(
        trie=trie,
        token_ids_to_sid=token_ids_to_sid,
        sid_mapping=mapping,
    )


def test_generative_diagnostics_cover_invalid_duplicate_seen_and_collision():
    decoder = _tiny_decoder()
    cases = [
        EvaluationCase(
            user_id=1,
            target_item_idx=1,
            generated_token_sequences=[
                [11, 12, 13, 14],
                [11, 12, 13, 14],
                [42, 43, 44, 45],
                [21, 22, 23, 24],
            ],
            seen_item_idx={0, 2},
        )
    ]

    report = compute_generative_diagnostics_from_cases(
        cases,
        decoder,
        top_k=2,
        beam_size=4,
        num_return_sequences=4,
        max_new_tokens=4,
        candidate_universe_size=3,
    )

    assert report["sid_diagnostics"]["invalid_sid_rate"] == pytest.approx(0.25)
    assert report["sid_diagnostics"]["valid_sid_rate"] == pytest.approx(0.75)
    assert report["sid_diagnostics"]["sid_collision_rate"] == pytest.approx(2 / 3)
    assert report["sid_diagnostics"]["unique_sid_ratio"] == pytest.approx(2 / 3)
    assert report["generation_diagnostics"]["duplicate_prediction_rate"] == pytest.approx(0.4)
    assert report["generation_diagnostics"]["seen_generated_rate"] == pytest.approx(0.75)
    assert report["generation_diagnostics"]["avg_raw_valid_candidates"] == pytest.approx(5.0)
    assert report["generation_diagnostics"]["avg_unique_filtered_candidates"] == pytest.approx(1.0)
    assert report["generation_diagnostics"]["insufficient_candidate_users"] == 1
    assert report["coverage"]["coverage@10"] == 1
    assert report["coverage"]["coverage@10_percent"] == pytest.approx(1 / 3)


def test_diagnostics_report_extracts_qwen_beam_sweep_fields(tmp_path):
    source = tmp_path / "beam_sweep_results.json"
    source.write_text(
        json.dumps(
            [
                {
                    "split": "test",
                    "beam_size": 10,
                    "num_return_sequences": 10,
                    "invalid_sid_rate": 0.2,
                    "seen_generated_rate": 0.3,
                    "raw_duplicate_valid_rate": 0.4,
                    "avg_raw_valid": 7.0,
                    "avg_unique_filtered_candidates": 5.0,
                    "coverage@10": 2,
                },
                {
                    "split": "test",
                    "beam_size": 20,
                    "num_return_sequences": 20,
                    "invalid_sid_rate": 0.1,
                    "coverage@10": 3,
                },
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "diagnostics.json"
    report = build_generative_diagnostics_report(
        input_path=source,
        output_path=output,
        run_id="tiny_qwen",
        trie_constrained=True,
        candidate_universe_size=10,
        selection=DiagnosticSelection(beam_size=10, num_return_sequences=10),
    )

    assert output.exists()
    assert report["run_id"] == "tiny_qwen"
    assert report["decoding"]["trie_constrained"] is True
    assert report["sid_diagnostics"]["invalid_sid_rate"] == pytest.approx(0.2)
    assert report["sid_diagnostics"]["valid_sid_rate"] == pytest.approx(0.8)
    assert report["generation_diagnostics"]["duplicate_prediction_rate"] == pytest.approx(0.4)
    assert report["generation_diagnostics"]["seen_generated_rate"] == pytest.approx(0.3)
    assert report["coverage"]["coverage@10_percent"] == pytest.approx(0.2)


def test_diagnostics_report_requires_explicit_selection_for_multi_record_input(tmp_path):
    source = tmp_path / "beam_sweep_results.json"
    source.write_text(
        json.dumps([{"beam_size": 10}, {"beam_size": 20}]),
        encoding="utf-8",
    )

    with pytest.raises(GenerativeDiagnosticsError, match="multiple records"):
        build_generative_diagnostics_report(
            input_path=source,
            output_path=tmp_path / "diagnostics.json",
        )


def test_diagnostics_report_cli_runs_on_tiny_input(tmp_path):
    source = tmp_path / "beam_sweep_results.jsonl"
    source.write_text(
        json.dumps({"split": "val", "beam_size": 5, "num_return_sequences": 5})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "diagnostics.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "plum_ml1m.cli",
            "diagnostics-report",
            "--input",
            str(source),
            "--output",
            str(output),
            "--run-id",
            "tiny_cli",
            "--trie-constrained",
        ],
        cwd=ROOT,
        env=_cli_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert report["run_id"] == "tiny_cli"
    assert report["split"] == "validation"


def test_committed_diagnostic_snapshots_have_expected_schema():
    paths = sorted(SNAPSHOT_DIR.glob("*.diagnostics.json"))
    assert paths

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert {"run_id", "split", "decoding", "sid_diagnostics"} <= set(data)
        assert {"generation_diagnostics", "coverage", "artifact_policy"} <= set(data)
        assert data["split"] in {"validation", "test"}
        assert {"invalid_sid_rate", "valid_sid_rate"} <= set(data["sid_diagnostics"])
        assert "avg_unique_filtered_candidates" in data["generation_diagnostics"]
        assert "coverage@10" in data["coverage"]
        assert data["artifact_policy"]["large_artifacts_committed"] is False
        assert data["artifact_policy"]["prediction_dumps_committed"] is False
