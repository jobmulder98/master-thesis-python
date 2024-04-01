from collections import defaultdict
from typing import List


def merge_dictionaries(dictionaries: List[dict]) -> dict:
    merged_dict = defaultdict(list)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged_dict[key].append(value)
    return dict(merged_dict)


