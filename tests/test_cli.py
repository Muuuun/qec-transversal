import json

from qec_transversal.cli import main


def test_cli_accepts_bit_strings(tmp_path) -> None:
    input_path = tmp_path / "steane.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(
        json.dumps(
            {
                "H_X": ["1111000", "1100110", "1010101"],
                "H_Z": ["1111000", "1100110", "1010101"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["analyze", str(input_path), "-o", str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["code"] == {"n": 7, "k": 1, "rank_X": 3, "rank_Z": 3}
    assert result["certificate"]["certified"] is True

