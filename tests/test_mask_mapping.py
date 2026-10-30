import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
import camvid_dataset as D  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "class_dict.csv"
        csv_path.write_text(
            "name,r,g,b\n"
            "Sky,128,128,128\n"
            "Road,128,64,128\n"
            "Car,64,0,128\n"
            "Void,0,0,0\n"
        )
        C.CLASS_DICT_CSV = csv_path
        C.IGNORE_CLASS_NAME = "Void"
        C.IGNORE_INDEX = 255
        m = D.build_label_mapping()

    assert m["num_labels"] == 3, m["num_labels"]
    assert m["id2label"] == {0: "Sky", 1: "Road", 2: "Car"}, m["id2label"]
    assert list(m["row_to_train"]) == [0, 1, 2, 255], list(m["row_to_train"])

    mask = np.array([[[128, 128, 128], [128, 64, 128]],
                     [[64, 0, 128], [0, 0, 0]]], dtype=np.uint8)
    idx = D.rgb_to_index(mask, m["palette"], m["row_to_train"])
    assert np.array_equal(idx, np.array([[0, 1], [2, 255]])), idx

    unknown = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert D.rgb_to_index(unknown, m["palette"], m["row_to_train"])[0, 0] == 255

    color = D.colorize(idx, m["train_palette"])
    assert np.array_equal(color[0, 0], [128, 128, 128])
    assert np.array_equal(color[1, 0], [64, 0, 128])
    assert np.array_equal(color[1, 1], [0, 0, 0])

    print("OK: class palette, train id numbering, RGB->index, ignore->255, colorize: all checks passed")


if __name__ == "__main__":
    main()
