# Total Perspective Vortex

## EEG Dataset

The dataset used was collected and contributed by Gerwin Schalk and his team from the **Brain-Computer Interface** (BCI) research program at the Wadsworth Center, New York State Department of Health.
The dataset contains over **1500 EEG** (electroencephalogram) recordings from **109 volunteers**.
These recordings were made as part of a study to understand brain activity associated with motor and imagery tasks, and the data were recorded using the BCI2000 system.
The dataset is publicly available through [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/S001/#files-panel), a respected resource for physiological signal research.

### Objectives

This dataset contains over 1500 EEG (electroencephalogram) recordings, each lasting one to two minutes, obtained from 109 volunteers. The data were collected to study brain activity related to various motor and imagery tasks. The recordings were made using the BCI2000 system, a sophisticated tool for capturing neural activity associated with real or imagined movements.

### Experimental Protocol

Participants underwent a series of 14 experimental sessions, which involved different motor and imagery tasks. The 14 sessions were conducted in the following order for each subject:

1. **Baseline, Eyes Open** (1 minute): Subjects remained still with their eyes open.
2. **Baseline, Eyes Closed** (1 minute): Subjects remained still with their eyes closed.
3. **Task 1** (2 minutes): Opening and closing the left or right fist in response to a target displayed on the screen.
4. **Task 2** (2 minutes): Imagining opening and closing the left or right fist in response to a target displayed on the screen.
5. **Task 3** (2 minutes): Opening and closing both fists or both feet depending on the position of a target on the screen (top or bottom).
6. **Task 4** (2 minutes): Imagining opening and closing both fists or both feet depending on the position of a target on the screen (top or bottom).
7. **Task 5** : Repeating Task 1.
8. **Task 6** : Repeating Task 2.
9. **Task 7** : Repeating Task 3.
10. **Task 8** : Repeating Task 4.
11. **Task 9** : Repeating Task 1.
12. **Task 10** : Repeating Task 2.
13. **Task 11** : Repeating Task 3.
14. **Task 12** : Repeating Task 4.

### Data Format

EEG recordings were made from 64 electrodes positioned according to the international 10-10 system, with a sampling rate of 160 Hz. The data are provided in EDF+ format (European Data Format), a standard for storing bioelectrical signal recordings.

Each data file includes annotations that mark significant events:
- **T0**: Rest periods.
- **T1**: Onset of real or imagined movement of the left hand or both hands.
- **T2**: Onset of real or imagined movement of the right hand or both feet.

These annotations are essential for analyzing brain activity specific to each task.
