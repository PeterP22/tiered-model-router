import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "plot_summary.py"
spec = importlib.util.spec_from_file_location("plot_summary", SCRIPT)
plot_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_summary)


def test_plot_reads_values_from_json_and_writes_png(tmp_path):
    results = {
        "n_test": 7,
        "v1_embed_lr": {"cpt_50": 0.11, "cpt_80": 0.22},
        "length_heuristic": {"cpt_50": 0.33, "cpt_80": 0.44},
        "random": {"cpt_50": 0.55, "cpt_80": 0.66},
    }
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(results))
    out = tmp_path / "out" / "chart.png"

    series = plot_summary.plot(eval_path, out)

    assert series == {
        "v1 router": [0.11, 0.22],
        "length heuristic": [0.33, 0.44],
        "random": [0.55, 0.66],
    }
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
