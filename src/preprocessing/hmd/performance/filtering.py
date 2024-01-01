import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import re

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import delta_time_seconds, pickle_exists, load_pickle

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
PERFORMANCE_FILENAME = os.getenv("PERFORMANCE_FILENAME")


def new_item_in_cart(dataframe: pd.DataFrame) -> list:
    new_item_in_cart_times = [0]
    item_added = dataframe["numberOfItemsInCart"].diff()
    new_items_df = dataframe.loc[item_added == 1, ["timeCumulative"]]
    new_item_in_cart_times.extend(new_items_df["timeCumulative"].tolist())
    return new_item_in_cart_times


def seconds_per_item(dataframe: pd.DataFrame) -> float:
    new_item_in_cart_times = new_item_in_cart(dataframe)
    return new_item_in_cart_times[-1] / (len(new_item_in_cart_times) - 1)


def remove_brackets_and_number(input_string):
    pattern = re.compile(r"\s*\(\d+\)$")
    result = re.sub(pattern, '', input_string)
    return result


def product_lists(dataframe: pd.DataFrame):
    condition_number = int(dataframe["condition"][0])
    df_correct_products = pd.read_excel(f"{DATA_DIRECTORY}/other/{PERFORMANCE_FILENAME}")
    correct_order = df_correct_products[f"c{condition_number}"].tolist()
    participant_picks = dataframe["itemsInCart"].iloc[-1]
    participant_picks = participant_picks.split(",")
    return correct_order, participant_picks


def count_errors(correct_order, participant_picks, participant, condition):
    errors = 0
    correct_index = 0
    participant_index = 0

    while correct_index < len(correct_order) and participant_index < len(participant_picks):
        correct_product = re.sub(r'\s+$', '', correct_order[correct_index].lower())
        participant_product = remove_brackets_and_number(participant_picks[participant_index]).lower()

        if correct_product == participant_product:
            correct_index += 1
            participant_index += 1
        else:
            errors += 1
            print(f"For participant {participant} in condition {condition}, the product is {correct_product}")
            next_correct_index = correct_index + 1 if correct_index + 1 < len(correct_order) else None
            next_next_correct_index = next_correct_index + 1 if (
                        next_correct_index is not None and next_correct_index + 1 < len(correct_order)) else None
            next_participant_index = participant_index + 1 if participant_index + 1 < len(participant_picks) else None

            next_correct_product = correct_order[next_correct_index].lower() if next_correct_index is not None else None
            next_next_correct_product = correct_order[next_next_correct_index].lower() if next_next_correct_index is not None else None
            next_participant_product = remove_brackets_and_number(participant_picks[next_participant_index]).lower() if next_participant_index is not None else None

            if next_correct_product == participant_product and next_participant_product == correct_product:
                # Swapped two products, skip next product
                correct_index += 2
                participant_index += 2
            elif next_correct_product == participant_product:
                # Participant skipped a product, move on to the next.
                correct_index += 1
            elif next_next_correct_product == participant_product:
                # Participant skipped two products, skip two products and add 1 error
                errors += 1
                correct_index += 2
            elif next_correct_product == next_participant_product:
                # Took incorrect product and moved on to the next one.
                correct_index += 1
                participant_index += 1
            else:
                # No matching products found, move on to the next participant product.
                participant_index += 1
    if participant_index < len(participant_picks):
        errors += len(participant_picks) - participant_index
    return errors


def n_back_performance_dataframe() -> pd.DataFrame:
    if pickle_exists("performance_dataframe.pickle"):
        plotting_dataframe = load_pickle("performance_dataframe.pickle")
    else:
        return pd.DataFrame()

    plotting_dataframe = plotting_dataframe[plotting_dataframe["condition"] == 7]
    n_back_dataframe = pd.read_excel(
        f"{DATA_DIRECTORY}/other/{PERFORMANCE_FILENAME}",
        sheet_name="correct-n-back-number-transpose"
    )
    given_answers = n_back_dataframe["total_correct"].iloc[0:22].tolist()
    plotting_dataframe["n_back_correct"] = given_answers
    return plotting_dataframe


# participant = 4  # 4, c5, 22, c5
# condition = 2
# df = create_clean_dataframe_hmd(participant, condition)
# cor, par = product_lists(df)
# errors = count_errors(cor, par, participant, condition)
# print(f"number of errors is {errors}")
