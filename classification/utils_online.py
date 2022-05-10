"""
Utility functions for the online scripts
"""

import os
import sys

import numpy as np

from subprocess import call
from tempfile import gettempdir

try:
    import winsound  # For Windows only
except:
    pass


def beep(waveform=(79, 45, 32, 50, 99, 113, 126, 127), win_freq=740):
    """
    Play a beep sound.

    Cross-platform sound playing with standard library only, no sound
    file required.

    From https://gist.github.com/juancarlospaco/c295f6965ed056dd08da
    """
    wavefile = os.path.join(os.getcwd(), "beep.wav")
    if not os.path.isfile(wavefile) or not os.access(wavefile, os.R_OK):
        with open(wavefile, "w+") as wave_file:
            for sample in range(0, 300, 1):
                for wav in range(0, 8, 1):
                    wave_file.write(chr(waveform[wav]))
    if sys.platform.startswith("linux"):
        return call("chrt -i 0 aplay '{fyle}'".format(fyle=wavefile), shell=1)
    if sys.platform.startswith("darwin"):
        return call("afplay '{fyle}'".format(fyle=wavefile), shell=True)
    if sys.platform.startswith("win"):  # FIXME: This is Ugly.
        winsound.Beep(win_freq, 500)
        return


def get_label_dict(info, n_class):
    """
    Get label names from stream info

    Parameters
    ----------

    info: LSL info object
        LSL Info object containing the label names.

    n_class: int
        Number of classes.

    Returns
    -------

    label_dict: dict
        Dictionary containing label info. Keys are label IDs and values are the label
        digit associated to them.
    """

    labels = info.desc().child("events_labels").child_value()
    label_list = [label for label in labels.split(",")]  # Formatting

    label_dict = {freq: idx for idx, freq in enumerate(set(label_list))}

    return label_dict


def get_channel_names(info):
    """
    Get channel names from stream info

    Parameters
    ----------

    info: LSL info object
        LSL Info object containing the label names.

    Returns
    -------

    ch_names: list
        Names of each channel, corresponding to the rows of the data
    """

    n_chan = info.channel_count()

    ch = info.desc().child("channels").first_child()
    ch_names = [ch.child_value("label")]

    for _ in range(n_chan - 1):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value("label"))

    return ch_names


def get_ch_idx(ch_names, ch_to_keep=None):
    """
    Return the indices of the desired elements of the list.

    Parameters
    ----------

    ch_names : list of str
        List containing the names of all the channels. These must
        be in the same order they are contained in the data array.

    ch_to_keep : list of str or None, default = None
        Channels to keep. If None, all channels are kept.

    Returns
    -------

    ch_idxs : list of int
        Indices of the channels to keep.
    """

    if not ch_to_keep:
        ch_to_keep = ch_names

    ch_dict = {ch: idx for idx, ch in enumerate(ch_names)}

    ch_idxs = [ch_dict.get(ch) for ch in ch_to_keep]

    return ch_idxs


def get_trial(
    data_inlet,
    marker_inlet,
    labels,
    event_id,
    ch_to_keep,
    epoch_len=1.4,
    delay=0.13,
    buffer_len=4,
    return_timestamps=True,
    sampling_mode="ms",
):
    """
    Get trigger-related data from the LSL data stream. The function never returns until
    a valid trial is completed. If a new event marker is received before a trial is completed,
    previous data is discarded in favor of the new trial. If a trigger corresponding to the
    current trial is received while data collection is in progress, it is interpreted as a
    cancel signal. In this case, the trial is dropped and the function waits for a new trial
    to re-start the process.

    Parameters
    ----------

    data_inlet: LSL StreamInlet
        The LSL stream that sends EEG data

    marker_inlet: LSL Stream Inlet
        LSL stream in charge of the event markers

    labels: list of str
        IDs of the triggers corresponding to the beginning of the trial

    buffer_len: float, default=4.
        Length of the data buffer in seconds

    return_timestamps: bool, default=True
        If True, return the list with timestamps for all samples
        of the trial

    Returns
    -------

    trial: np.array of shape (n_channels, n_samples)
        EEG data corresponding to the trial

    label: int
        Label corresponding to the trial

    epoch_times: list of float
        List containing the timestamps associated with the
        sending time of each sample of the trial. Only returned
        if return_timestamps=True
    """

    # Data parameters
    n_chan = data_inlet.info().channel_count()
    sfreq = int(data_inlet.info().nominal_srate())

    # Buffer to deque incoming training data
    data_buffer = np.zeros((n_chan, sfreq * buffer_len)) - 10
    _, buffer_samples = data_buffer.shape

    # Buffer to deque incoming timestamps
    times_buffer = np.zeros((buffer_samples)) - 10

    # Target time initialization
    target_time = np.inf

    # How much samples to collect
    samp = 10

    got_marker = False

    while True:
        # Pull data in small chunks
        eeg_data, data_times = data_inlet.pull_chunk(
            timeout=1 / (2 * sfreq), max_samples=samp
        )

        if eeg_data:
            # Prepare the data
            eeg_array = np.array(eeg_data).T
            times_array = np.array(data_times)

            if sampling_mode == "ms":
                times_array = np.round(times_array, 3)  # In miliseconds

            # Deque data and times arrays
            data_buffer = np.hstack((data_buffer, eeg_array))
            data_buffer = data_buffer[..., -buffer_samples:]

            times_buffer = np.hstack([times_buffer, times_array])

            times_buffer = times_buffer[-buffer_samples:]

            # Check if there is an event marker
            if got_marker == False:
                marker, marker_time = marker_inlet.pull_sample(timeout=0.0)
                if marker:
                    label = marker[0].split(",")[0]
                    if label in labels:
                        marker_time = np.array(marker_time)
                        got_marker = True

            if got_marker:
                # print(times_array)
                # Get more samples per pull_chunk
                # as we are not waiting for a marker anymore
                samp = int(sfreq * epoch_len / 4.0)

                # Store label and timestamp
                true_label = event_id[label]

                if sampling_mode == "ms":
                    marker_time = np.round(marker_time, 3)
                # Parameter to modify the calculation of target time
                if sampling_mode == "ms":
                    time_mod = 1  # Timestamps from LiveAmp come already in miliseconds
                elif sampling_mode == "samples":
                    time_mod = sfreq

                # Find your target time
                total_len = (
                    epoch_len * time_mod + delay
                )  # epoch_len is in s so we convert to ms
                target_time = np.round(marker_time + total_len, 3)  # In samples

            if times_buffer[-1] > target_time:

                print(
                    f"Marker time for the beginning of the epoch is: {marker_time + delay}"
                )
                print(f"Target timestamp for the end of this epoch is: {target_time}")
                print("")

                # Find the index of the first and last sample
                first_sample = np.where(times_buffer >= marker_time + delay)[0][0]
                last_sample = int(first_sample + (sfreq * epoch_len))

                if last_sample < buffer_samples - 1:
                    print("First sample", first_sample, "last sample", last_sample)
                    while last_sample - first_sample != epoch_len * sfreq:
                        print("Another one")
                        last_sample += 1
                    # Keep only the channels we are interested in
                    data_buffer = data_buffer[ch_to_keep, :]
                    # Slice the thing
                    epoch = data_buffer[:, first_sample:last_sample]

                    epoch_times = times_buffer[first_sample:last_sample]
                    # Average re-referencing
                    ref_data = epoch.mean(0, keepdims=True)
                    epoch -= ref_data

                    # Baseline correction
                    mean = np.mean(epoch, axis=1, keepdims=True)
                    epoch -= mean

                    # Reset target time
                    target_time = np.inf

                    # Reset number of samples per pull_chunk
                    samp = 10
                    got_marker = False

                    if return_timestamps:
                        return epoch, true_label, epoch_times
                    else:
                        return epoch, true_label, None
