import json
import os

from causalforge.cli import main


def test_cli_benchmark(monkeypatch, tmp_path):
    monkeypatch.setenv("CAUSALFORGE_N_SAMPLES", "400")
    monkeypatch.setenv("CAUSALFORGE_N_REPS", "4")
    out = str(tmp_path / "bench.json")
    rc = main(["benchmark", "--out", out, "--seed", "5"])
    assert rc == 0
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert len(payload["results"]) >= 5


def test_cli_run(tmp_path):
    out = str(tmp_path / "res.json")
    rc = main(["run", "--seed", "5", "--n", "400", "--out", out])
    assert rc == 0
    assert os.path.exists(out)


def test_cli_version():
    rc = main(["version"])
    assert rc == 0
