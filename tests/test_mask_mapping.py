"""
Синтетический тест конвертации масок — без torch, только numpy/PIL.
Запуск:  python3 tests/test_mask_mapping.py
Проверяет: чтение class_dict.csv, нумерацию train-id, Void->255,
корректный RGB->index и обратную раскраску.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402


def main():
    tmp = ROOT / "tests" / "_tmp_class_dict.csv"
    tmp.write_text(
        "name,r,g,b\n"
        "Sky,128,128,128\n"
        "Road,128,64,128\n"
        "Car,64,0,128\n"
        "Void,0,0,0\n"
    )
    # направляем конфиг на синтетический csv и включаем игнор Void
    C.CLASS_DICT_CSV = tmp
    C.IGNORE_CLASS_NAME = "Void"
    C.IGNORE_INDEX = 255

    import camvid_dataset as D

    m = D.build_label_mapping()

    # Void исключён из num_labels -> 3 обучаемых класса
    assert m["num_labels"] == 3, m["num_labels"]
    assert m["id2label"] == {0: "Sky", 1: "Road", 2: "Car"}, m["id2label"]
    # Void -> 255, остальные -> 0,1,2 (строки csv: Sky,Road,Car,Void)
    assert list(m["row_to_train"]) == [0, 1, 2, 255], list(m["row_to_train"])

    # собираем маску 2x2: Sky, Road / Car, Void
    mask = np.array([[[128, 128, 128], [128, 64, 128]],
                     [[64, 0, 128],   [0, 0, 0]]], dtype=np.uint8)
    idx = D.rgb_to_index(mask, m["palette"], m["row_to_train"])
    expected = np.array([[0, 1], [2, 255]], dtype=np.int64)
    assert np.array_equal(idx, expected), idx

    # неизвестный цвет -> IGNORE_INDEX
    unknown = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert D.rgb_to_index(unknown, m["palette"], m["row_to_train"])[0, 0] == 255

    # обратная раскраска: валидные пиксели вернули исходный цвет, Void -> чёрный
    color = D.colorize(idx, m["train_palette"])
    assert np.array_equal(color[0, 0], [128, 128, 128])   # Sky
    assert np.array_equal(color[1, 0], [64, 0, 128])      # Car
    assert np.array_equal(color[1, 1], [0, 0, 0])         # Void (ignored) -> чёрный

    tmp.unlink()
    print("OK: class_dict, train-id нумерация, RGB->index, Void->255, раскраска — все проверки пройдены")


if __name__ == "__main__":
    main()
