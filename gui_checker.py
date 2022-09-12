import os
import pandas as pd
import numpy as np
import PySimpleGUI as sg

def get_data_for_encounter(encounter_db_line):
    # Get middle timestamp
    center = np.mean([encounter_db_line.begin, encounter_db_line.end])

    # Scrape directory
    camera_folder = os.path.join('H:', 'bagfiles_unpack', encounter_db_line.bag_file, 'camera_0')
    frame_list = os.listdir(camera_folder)
    frame_list_float = [float(frame[:-4]) for frame in frame_list]
    closest_index = np.argmin(abs(frame_list_float - center))
    frame_to_use = frame_list[closest_index]
    second_frame_to_use = frame_list[closest_index + 1]

    return os.path.join(camera_folder, frame_to_use), os.path.join(camera_folder, second_frame_to_use), encounter_db_line.description, encounter_db_line.is_encounter, encounter_db_line.direction, encounter_db_line.distance


# Open feather file
encounter_db = pd.read_feather('H:/bagfiles_unpack/encounter_db_v2_backup_after_manual.feather')
encounter_db = encounter_db.drop_duplicates(subset=['begin', 'end', 'description']).reset_index().drop('index', axis=1)

# Start index of unchecked data
first_unchecked = min(encounter_db[encounter_db.manual_override == False].index)

# Get first entry
image_path_0, image_path_1, description, is_encounter, direction, distance = get_data_for_encounter(encounter_db.iloc[first_unchecked])

# Counter to go through all encounters
counter = first_unchecked + 1


sg.theme('DarkAmber')   # Add a touch of color

# All the stuff inside your window.
layout = [  [sg.Image(image_path_0, key="-IMAGE-")],
            [sg.Text(f"Direction: {direction}, Description: {description}, Distance: {distance}", key="-TEXT-")],
            [ sg.Button('Oppose'), sg.Button('None'), sg.Button('Overtake'), sg.VerticalSeparator(), sg.Button('Prev'), sg.Button('Next'), sg.VerticalSeparator(), sg.Button('GO BACK')] ]

# Create the Window
window = sg.Window('GUI Checker', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()

    # If user closes window
    if event == sg.WIN_CLOSED or counter == len(encounter_db):
        break

    if event == 'No':
        print(0, -1, int(is_encounter), direction)
        encounter_db.at[counter - 1, 'is_encounter'] = False
        encounter_db.at[counter - 1, 'direction'] = -1
        encounter_db.at[counter - 1, 'manual_override'] = True

        # Get next encounter
        image_path_0, image_path_1, description, is_encounter, direction, distance = get_data_for_encounter(encounter_db.iloc[counter])
        counter += 1

        window["-IMAGE-"].update(image_path_0)
        window["-TEXT-"].update(f"Direction: {direction}, Description: {description}, Distance: {distance}")

    elif event == 'Overtake':
        print(1, 1, int(is_encounter), direction)
        encounter_db.at[counter - 1, 'is_encounter'] = True
        encounter_db.at[counter - 1, 'direction'] = 1
        encounter_db.at[counter - 1, 'manual_override'] = True

        # Get next encounter
        image_path_0, image_path_1, description, is_encounter, direction, distance = get_data_for_encounter(encounter_db.iloc[counter])
        counter += 1

        window["-IMAGE-"].update(image_path_0)
        window["-TEXT-"].update(f"Direction: {direction}, Description: {description}, Distance: {distance}")

    elif event == 'Oppose':
        print(1, 0, int(is_encounter), direction)
        encounter_db.at[counter - 1, 'is_encounter'] = True
        encounter_db.at[counter - 1, 'direction'] = 0
        encounter_db.at[counter - 1, 'manual_override'] = True

        # Get next encounter
        image_path_0, image_path_1, description, is_encounter, direction, distance = get_data_for_encounter(encounter_db.iloc[counter])
        counter += 1

        window["-IMAGE-"].update(image_path_0)
        window["-TEXT-"].update(f"Direction: {direction}, Description: {description}, Distance: {distance}")

    elif event == 'Next':
        window["-IMAGE-"].update(image_path_1)

    elif event == 'Prev':
        window["-IMAGE-"].update(image_path_0)

    elif event == 'GO BACK':
        counter -= 1
        image_path_0, image_path_1, description, is_encounter, direction, distance = get_data_for_encounter(encounter_db.iloc[counter - 1])

        window["-IMAGE-"].update(image_path_0)
        window["-TEXT-"].update(f"Direction: {direction}, Description{description}, Distance: {distance}")

window.close()
#encounter_db.to_feather('H:/bagfiles_unpack/encounter_db_v2.feather')