from pathlib import Path

import pytest

from tom_research.study import load_study, prepare_pilot


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("track,answer_format", [("thoughttracing", "free_text"), ("autotom", "multiple_choice")])
def test_tracks_have_separate_sources_and_methods(tmp_path, track, answer_format):
    from tom_research.study import TRACKS

    source = TRACKS[track]["source"]
    manifest = tmp_path / f"{track}.json"
    prepare_pilot(source, manifest, track=track)
    debug, pilot, metadata = load_study(source, manifest, track=track)
    assert metadata["track"] == track
    assert metadata["methods"] == ["plain", track]
    assert metadata["answer_format"] == answer_format
    assert all(item.answer_format == answer_format for item in debug + pilot)
    assert len(pilot) == 20
    with pytest.raises(ValueError, match="track"):
        load_study(source, manifest, track="autotom" if track == "thoughttracing" else "thoughttracing")


def test_frozen_pilot_is_balanced_and_reloads(tmp_path):
    source = ROOT / "external/thought-tracing/data/paraphrased_tomi/test.txt"
    if not source.exists():
        pytest.skip("Fetch official sources first")
    manifest = tmp_path / "pilot_ids.json"
    prepare_pilot(source, manifest)
    debug, pilot, metadata = load_study(source, manifest)
    assert len(debug) == 2 and len(pilot) == 20
    assert sum(item.belief_type == "true_belief" for item in pilot) == 10
    assert len({item.story_id for item in debug + pilot}) == 22
    assert metadata["answer_format"] == "free_text"
    before = manifest.read_text()
    prepare_pilot(source, manifest)
    assert before == manifest.read_text()


def test_refuses_changed_source(tmp_path):
    import json

    source = tmp_path / "test.txt"
    source.write_text("changed")
    source.with_suffix(".trace").write_text("changed")
    manifest = tmp_path / "pilot_ids.json"
    manifest.write_text(json.dumps({"source_sha256": "wrong", "trace_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash"):
        load_study(source, manifest)