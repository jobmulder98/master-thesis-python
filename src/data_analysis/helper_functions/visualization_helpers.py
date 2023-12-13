import numpy as np


def increase_opacity_condition(outline_condition, number_of_conditions):
    if outline_condition:
        if isinstance(outline_condition, int):
            opacity = np.ones(number_of_conditions) * 0.2
            opacity[outline_condition - 1] = 1
        elif isinstance(outline_condition, list):
            opacity = np.ones(number_of_conditions) * 0.2
            for condition in outline_condition:
                opacity[condition - 1] = 1
        else:
            opacity = np.ones(number_of_conditions)
    else:
        opacity = np.ones(number_of_conditions)
    return opacity
