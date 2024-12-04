DATASET_PATH = "/home/cgangaro/goinfre/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/"
MODELS_PATH = "./models/"

amount_of_subjects = 109
amount_of_runs = 14
batch_read = 50

plotting = False

good_channels = ["C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6"]

csp_params = {
    "n_components": 4
}

test_size = 0.2

experiments = [
    {
        "name": "Left_right_fist",
        "description": "open and close left or right fist",
        "runs": [3, 7, 11],
        "mapping": {
            "0": "Rest",
            "1": "Left fist",
            "2": "Right fist"
        },
    },
    {
        "name": "Imagine_left_right_fist",
        "description": "imagine opening and closing left or right fist",
        "runs": [4, 8, 12],
        "mapping": {
            0: "Rest",
            1: "Imagine left fist",
            2: "Imagine right fist"
        },
    },
    {
        "name": "Fists_feet",
        "description": "open and close both fists or both feet",
        "runs": [5, 9, 13],
        "mapping": {
            0: "Rest",
            1: "Both fists",
            2: "Both feet"
        },
    },
    {
        "name": "Imagine_fists_feet",
        "description": "imagine opening and closing both fists or both feet",
        "runs": [6, 10, 14],
        "mapping": {
            0: "Rest",
            1: "Imagine both fists",
            2: "Imagine both feet"
        },
    },
    {
        "name": "Movement_of_fists",
        "description": "movement (real or imagined) of fists",
        "runs": [3, 7, 11, 4, 8, 12],
        "mapping": {
            0: "Rest",
            1: "Left fist",
            2: "Right fist"
        },
    },
    {
        "name": "Movement_fists_feet",
        "description": "movement (real or imagined) of fists or feet",
        "runs": [5, 9, 13, 6, 10, 14],
        "mapping": {
            0: "Rest",
            1: "Both fists",
            2: "Both feet"
        },
    }
]
