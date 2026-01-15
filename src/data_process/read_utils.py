import json
import pandas as pd
import os
from typing import List, Tuple, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "MLA_100k_checked_v3.jsonlines")
print(f"Data path: {DATA_PATH}")
# You can safely assume that `build_dataset` is correctly implemented
def build_dataset(data_path: str = DATA_PATH) -> Tuple[List[Dict], List, List[Dict], List]:
    data = [json.loads(x) for x in open(data_path)]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def flatten_items(items: List[Dict]) -> pd.DataFrame:
    """
    Convierte una lista de dicts anidados en un DataFrame plano.
    """
    flat_items = [flatten_dict(item) for item in items]
    return pd.DataFrame(flat_items)