import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["00_data.ipynb", "01_debug.ipynb", "02_pilot.ipynb"])
@pytest.mark.parametrize("track", ["thoughttracing", "autotom"])
def test_notebook_runs_offline_from_fresh_kernel(name, tmp_path, track):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    if not (ROOT / "external/thought-tracing/data/paraphrased_tomi/test.txt").exists():
        pytest.skip("Fetch pinned sources with scripts/bootstrap.sh")
    client = NotebookClient(notebook, timeout=90, kernel_name="tom-research",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute(env={**os.environ, "TOM_RESULTS_DIR": str(tmp_path), "TOM_TRACK": track})
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert all(output.output_type != "error" for output in cell.get("outputs", []))
    if name == "02_pilot.ipynb":
        import json
        results = [json.loads(line) for line in (tmp_path / track / "pilot/predictions.jsonl").read_text().splitlines()]
        assert len(results) == 40
        assert {record["method"] for record in results} == {"plain", track}
        assert all(record["track"] == track for record in results)
        assert all(record["status"] == "synthetic" and record["usage"]["calls"] == 0 for record in results)