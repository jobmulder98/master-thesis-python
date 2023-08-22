from datetime import datetime
import pytz
import pyautogui
from pynput.mouse import Listener
import time

# coordinates on Job's laptop
# region_left = 721
# region_top = 593
# region_right = 1036
# region_bottom = 624

# TODO coordinates on TU Delft computer
region_left = 0
region_top = 0
region_right = 0
region_bottom = 0

output_file_path = r"C:\Users\jobmu\OneDrive\Documenten\Master jaar 2\Master Thesis\Master Thesis\Experiment\biotrace-timestamps.txt"


def find_screen_coordinates(x, y, button, pressed):
    if pressed:
        x, y = pyautogui.position()
        print(x, ",", y)
        return


def record_timestamp_on_click(x, y, button, pressed):
    if pressed:
        x, y = pyautogui.position()
        if region_left <= x <= region_right and region_top <= y <= region_bottom:
            timezone = pytz.timezone("Europe/Amsterdam")
            current_time = datetime.now(timezone)
            current_time_reformatted = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            timestamp = time.time()
            with open(output_file_path, 'a') as file:
                file.write(f"{current_time_reformatted}, {timestamp} \n")
            print(f"Timestamp added to the file.")
            return


with Listener(on_click=record_timestamp_on_click) as listener:
    listener.join()
