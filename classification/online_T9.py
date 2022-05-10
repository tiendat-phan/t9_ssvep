"""
Script to read and classify EEG data in real time.

Authors: udovic Darmet, Juan Jesus Torre Tresols 
Mail: ludovic.darmet@siae-supaero.fr; Juan-jesus.TORRE-TRESOLS@isae-supaero.fr
"""

import argparse
from math import inf
import os
import pickle

import sys
from turtle import down

import numpy as np
import pandas as pd

from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    resolve_byprop,
    local_clock,
)
from subprocess import call

from TRCA import TRCA

from utils_online import beep, get_label_dict, get_ch_idx, get_channel_names, get_trial


parser = argparse.ArgumentParser(description="Parameters for the experiment")
parser.add_argument(
    "-e",
    "--epochlen",
    metavar="EpochLength",
    type=float,
    default=2.0,
    help="Length of each data epoch used for " "classification. Default: %(default)s.",
)
parser.add_argument(
    "-b",
    "--buffer",
    metavar="BufferLength",
    type=int,
    default=4.0,
    help="Length of the data buffer in seconds. " "Default: %(default)s.",
)
parser.add_argument(
    "-ds",
    "--datastream",
    metavar="DataStream",
    type=str,
    default="SimulatedData",
    help="Name of the data stream to look for",
)
parser.add_argument(
    "-ms",
    "--markerstream",
    metavar="MarkerStream",
    type=str,
    default="MyMarkerStream",
    help="Name of the marker stream to look for",
)
parser.add_argument(
    "-m",
    "--mode",
    metavar="SampleMode",
    type=str,
    default="ms",
    choices=["samples", "ms"],
    help="Format for the event timestamps. Can be samples or miliseconds. "
    "Default: %(default)s. Choices: %(choices)s",
)

args = parser.parse_args()

## Argparse Parameters
buffer_len = (
    args.buffer
)  # Length of the array that keeps the data stored from the stream (s)
epoch_len = args.epochlen  # Length of each data epoch used as observations
data_name = args.datastream
marker_name = args.markerstream
sampling_mode = args.mode

## LSL streams
# Create outlet for clf signal
clf_info = StreamInfo(
    name="TRCAOutput",
    type="TRCA",
    channel_count=1,
    nominal_srate=500.0,
    channel_format="int8",
    source_id="coolestIDever1234",
)

clf_outlet = StreamOutlet(clf_info)

# First resolve a data stream
print("Looking for a data stream...")
data_streams = resolve_byprop("type", "EEG", timeout=5)

# If nothing is found, raise an error
if len(data_streams) == 0:
    raise (RuntimeError("Can't find EEG stream..."))
else:
    print("Data stream found!")

# Then resolve the marker stream
print("Looking for a marker stream...")
marker_streams = resolve_byprop("name", marker_name, timeout=120)

# If nothing is found, raise an error
if len(marker_streams) == 0:
    raise (RuntimeError("Can't find marker stream..."))
else:
    print("Marker stream found!")

# Get data inlet
data_inlet = StreamInlet(
    data_streams[0], max_buflen=10, max_chunklen=1, processing_flags=1
)  # max_buflen should be in s
marker_inlet = StreamInlet(marker_streams[0], max_chunklen=1, processing_flags=1)

# Get the stream info and description
marker_info = marker_inlet.info()
data_info = data_inlet.info()

description = marker_info.desc()

## Parameters
buffer_len = (
    args.buffer
)  # Length of the array that keeps the dnamesata stored from the stream (s)
epoch_len = args.epochlen  # Length of each data epoch used as observations
sfreq = int(data_info.nominal_srate())
if sfreq >= 500:
    dowsample = 2
else:
    downsample = 1
delay = int(sfreq * 0.135)  # In samples
if sampling_mode == "ms":
    delay *= 1 / sfreq  # In ms
n_chan = data_info.channel_count()
n_samples = int(sfreq * epoch_len)  # Number of samples per epoch

ch_slice = ["O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8"]  # Channels to keep
ch_slice = ["13", "14", "15", "16", "17", "18", "19", "20"]

## CLF parameters
n_classes = int(description.child("n_class").child_value())  # Number of classes
n_train = int(
    description.child("n_train").child_value()
)  # Calibration trials per class
labels = description.child(
    "events_labels"
).child_value()  # List containing all the stim triggers
cues = description.child(
    "cues_labels"
).child_value()  # List containing all the cue triggers
filename = description.child(
    "filename"
).child_value()  # String containing participant and session number
amp = description.child(
    "amp"
).child_value()  # String containing amplitude of the stimuli
cal_trials = n_train * n_classes

labels = [label for label in labels.split(",")]
print("Labels", labels)
cues = [cue for cue in cues.split(",")]

event_id = get_label_dict(marker_info, n_classes)
ch_names = get_channel_names(data_info)
print(ch_names)

ch_to_keep = get_ch_idx(ch_names, ch_slice)
print(f"\n Channels number to keep: {ch_to_keep} \n")

# Parameters for the LSL processing
data_buffer_len = 4  # In seconds

peaks = [float(key.split("_")[0]) for key in event_id.keys()]
if np.max(peaks) < 20:
    nfbands = 5
else:
    nfbands = 2

if np.max(peaks) > 20:
    cond = "_high_"
else:
    cond = "_low_"

amp = "_amp" + amp + "_"

## Load or create model
model_filename = os.path.join(
    os.getcwd(), filename + cond + amp + "TRCA_calibration.sav"
)
caldata_filename = os.path.join(
    os.getcwd(), filename + cond + amp + "calibration_data.npy"
)
trustscore_filename = os.path.join(os.getcwd(), filename + cond + amp + "_scores.csv")

try:
    X_train, y_train = pickle.load(open(caldata_filename, "rb"))
    clf = TRCA(
        sfreq=sfreq * 1.0,
        peaks=peaks,
        downsample=downsample,
        n_fbands=nfbands,
        method="original",
        regul="lwf",
        trustscore=False,
    )
    clf.fit(X_train, y_train)
    model_loaded = True

    print(f"Using saved data - {model_filename}")
    print("")

except FileNotFoundError:
    clf = TRCA(
        sfreq=sfreq * 1.0,
        peaks=peaks,
        downsample=downsample,
        n_fbands=nfbands,
        method="original",
        regul="lwf",
        trustscore=False,
    )
    model_loaded = False

    print("Calibration file not found...")
    print("")

## Skip calibration if model was found
if not model_loaded:
    ## CALIBRATION
    print("")
    print("-" * 21)
    print("Starting calibration")
    print("-" * 21)
    print("")

    print(f"Expected number of classes: {n_classes}")
    print(f"Expected number of calibration trials (per class): {n_train}")
    print("")
    print(f"Expected number of calibration trials (total): {cal_trials}")
    print("")

    X_train = np.zeros((cal_trials, len(ch_to_keep), n_samples))
    y_train = []

    # Pause the execution to set up what you need. Unpause with Intro key
    # print("Ready to start calibration, press the 'Intro' key to start...\n")

    # Number of training trial
    training_idx = 0

    # Get time 0 to correct the timestamps
    t0 = local_clock()

    while training_idx < cal_trials:
        print(training_idx)
        # Get training trial
        cal_trial, true_label, epoch_times = get_trial(
            data_inlet,
            marker_inlet,
            labels,
            event_id,
            ch_to_keep,
            epoch_len,
            delay,
            buffer_len=data_buffer_len,
            sampling_mode=sampling_mode,
        )

        # Add the epoch to the training data with its corresponding label
        X_train[training_idx, :, :] = cal_trial
        y_train.append(true_label)

        print(f"Start and end of the epoch: {epoch_times[0]}, {epoch_times[-1]}")
        print(f"Correctly stored calibration trial n {training_idx + 1}")
        print("")

        target_time = 0
        training_idx += 1

    print(f"Calibration data recorded. Final shape of X_train: {X_train.shape}")
    pickle.dump((X_train, y_train), open(caldata_filename, "wb"))

    print("")
    print("-" * 21)
    print("Fitting training data...")
    print("-" * 21)
    print("")

    clf.fit(X_train, y_train)

    print("Data was fit. Calibration complete")
    print("")

    ## Save the model if calibration was done
    pickle.dump(clf, open(model_filename, "wb"))

    print("")
    print(f"Model saved in {model_filename}")

prediction = []
test_idx = 0

## TESTING
# Pause the execution to set up what you need. Unpause with Intro key
# print("Ready to start testing, press the 'Intro' key to start...\n")
outputs = {"y_pred": [], "y_true": []}
while True:
    # Get test trial
    X_test, true_label, epoch_times = get_trial(
        data_inlet,
        marker_inlet,
        labels,
        event_id,
        ch_to_keep,
        epoch_len,
        delay,
        buffer_len=data_buffer_len,
        sampling_mode=sampling_mode,
    )

    # Predict on your data and check if it is correct
    y_pred = clf.predict(X_test)
    for k, v in event_id.items():
        if v == y_pred[0]:
            pred = k.split("_")[-1]
    for k, v in event_id.items():
        if v == true_label:
            true = k.split("_")[-1]
    if pred == "Back":
        pred = 10
    else:
        pred = int(pred)
    if true == "Back":
        true = 10
    else:
        true = int(true)

    outputs["y_pred"].append(pred)
    outputs["y_true"].append(true)

    clf_outlet.push_sample([pred])

    print(f"Start and end of the epoch: {epoch_times[0]}, {epoch_times[-1]}")
    print("")

    if y_pred[0] == true_label:
        prediction.append(1)
        print("Correct prediction!")
        beep()

    else:
        last_test = X_test
        prediction.append(0)
        print("Booooh")
        beep(win_freq=440)

    print(f"Predicted label: {pred}, True_label: {true}")
    print("-" * 20)
    print("")

    target_time = 0
    test_idx += 1

    df = pd.DataFrame(outputs)
    df.to_csv(trustscore_filename, index=None)
    print(f"Clf score: {sum(prediction) / len(prediction)}")
