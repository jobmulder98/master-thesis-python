import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import re

from src.preprocessing.hmd.clean_raw_data import create_clean_dataframe_hmd
from src.preprocessing.helper_functions.general_helpers import delta_time_seconds

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")


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
    df_correct_products = pd.read_excel(f"{DATA_DIRECTORY}/other/performance-results.xlsx")
    correct_order = df_correct_products[f"c{condition_number}"].tolist()
    participant_picks = dataframe["itemsInCart"].iloc[-1]
    participant_picks = participant_picks.split(",")
    return correct_order, participant_picks


def count_errors(correct_order, participant_picks):
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
            print(correct_product)
            next_correct_index = correct_index + 1 if correct_index + 1 < len(correct_order) else None
            next_participant_index = participant_index + 1 if participant_index + 1 < len(participant_picks) else None

            next_correct_product = correct_order[next_correct_index].lower() if next_correct_index is not None else None
            next_participant_product = remove_brackets_and_number(participant_picks[next_participant_index]).lower() if next_participant_index is not None else None

            if next_correct_product == participant_product:
                # Participant skipped a product, move on to the next.
                correct_index += 1
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


participant = 10
condition = 4
df = create_clean_dataframe_hmd(participant, condition)
cor = ['Milk', 'Bread', 'Basil', 'Dough blue', 'Bottle coke', 'Bottle', 'Chips red', 'Beerbox rood', 'Tomato', 'Banana', 'Cheese ', 'SprayBottle', 'Soil', 'Bottle fanta', 'Joghurt yellow', 'Carrot', 'BeerBox green', 'Joghurt purple', 'Onion', 'Joghurt', 'Chips white', 'Broccoli', 'Meat', 'Broccoli', 'Pepperoni', 'Cheese white', 'Lemon', 'Apple', 'Pear', 'BeerBox green']
par = ['Milk', 'Bread (6)', 'Basil', 'Dough Blue (1)', 'Bottle coke (6)', 'Bottle (2)', 'Chips red (4)', 'BeerBox rood', 'Tomato (7)', 'banana (2)', 'Cheese', 'SprayBottle (6)', 'Soil (5)', 'Bottle fanta', 'Joghurt yellow', 'Carrot (2)']

print(count_errors(cor, par))