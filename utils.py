import json
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))


class DataPaths:
    def __init__(self, base_path: Optional[str] = None) -> None:
        if base_path is None:
            base_path = os.path.join(PROJECT_ROOT_PATH, "data")
        self.base_path = base_path

        self.elibrary_oecd_full_train = os.path.join(base_path, "elibrary_oecd_full", "train.csv")
        self.elibrary_oecd_full_test = os.path.join(base_path, "elibrary_oecd_full", "test.csv")

        self.elibrary_grnti_full_train = os.path.join(
            base_path, "elibrary_grnti_full", "train.csv"
        )
        self.elibrary_grnti_full_test = os.path.join(base_path, "elibrary_grnti_full", "test.csv")

        self.elibrary_oecd_ru_train = os.path.join(base_path, "elibrary_oecd_ru", "train.csv")
        self.elibrary_oecd_ru_test = os.path.join(base_path, "elibrary_oecd_ru", "test.csv")

        self.elibrary_grnti_ru_train = os.path.join(base_path, "elibrary_grnti_ru", "train.csv")
        self.elibrary_grnti_ru_test = os.path.join(base_path, "elibrary_grnti_ru", "test.csv")

        self.elibrary_oecd_en_train = os.path.join(base_path, "elibrary_oecd_en", "train.csv")
        self.elibrary_oecd_en_test = os.path.join(base_path, "elibrary_oecd_en", "test.csv")

        self.elibrary_grnti_en_train = os.path.join(base_path, "elibrary_grnti_en", "train.csv")
        self.elibrary_grnti_en_test = os.path.join(base_path, "elibrary_grnti_en", "test.csv")

        self.ru_en_translation_test = os.path.join(base_path, "ru_en_translation_test.json")


def load_embeddings_from_jsonl(embeddings_path: str) -> dict[str, np.array]:
    """Load embeddings from a jsonl file.
    The file must have one embedding per line in JSON format.
    It must have two keys per line: `paper_id` and `embedding`

    Arguments:
        embeddings_path -- path to the embeddings file

    Returns:
        embeddings -- a dictionary where each key is the paper id
                                   and the value is a numpy array
    """
    embeddings = {}
    with open(embeddings_path, "r") as f:
        for line in tqdm(f, desc="reading embeddings from file..."):
            line_json = json.loads(line)
            embeddings[line_json["paper_id"]] = np.array(line_json["embedding"])
    return embeddings


def print_metrics(task_name: str, results: dict, silent: bool = False) -> None:
    if silent:
        return

    print("-" * 30)
    for metric_name, value in results[task_name].items():
        if metric_name != "cls_report":
            print(f"{task_name} | {metric_name} | = {round(value, 2)}")
    print("-" * 30)
