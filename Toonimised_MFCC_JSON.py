# Audiofailide toonimine ja MFCC JSON-ite loomine
# Raivo Kasepuu
# B710710
# 13.04.2023

import json
import os
import math
import librosa
import time
import numpy as np
import torch
import torchaudio.transforms as T


# Konstandid
NUM_MFCC = 26
NUM_SEGMENTS = 10
HOP_LENGTH = 512
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Sisend ja väljundkaustad
PATH = "/Users/raivo/Documents/Magister_files/augJSON"
DATASET_PATH = "/Users/raivo/Documents/Magister_files/Data/genres_original"
JSON_FOLDER = PATH + str(NUM_SEGMENTS) + "_" + str(NUM_MFCC) + "/"

augmentation_dicts = [
    {'def_name': 'original', 'arg_1': None, 'arg_2': None},
    {'def_name': 'polarity_inversion', 'arg_1': None, 'arg_2': None},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': 1},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': 2},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': 3},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': -1},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': -2},
    {'def_name': 'pitch_shift', 'arg_1': None, 'arg_2': -3},
    {'def_name': 'pre_emphasis', 'arg_1': None, 'arg_2': 0.99},
    {'def_name': 'pre_emphasis', 'arg_1': None, 'arg_2': 0.98},
    {'def_name': 'pre_emphasis', 'arg_1': None, 'arg_2': 0.97},
    {'def_name': 'dynamic_range_compression', 'arg_1': None, 'arg_2': 1},
    {'def_name': 'dynamic_range_compression', 'arg_1': None, 'arg_2': 2},
    {'def_name': 'dynamic_range_compression', 'arg_1': None, 'arg_2': 3},
]


def original(signal, y):
    return signal


def polarity_inversion(signal, y):
    return -signal


def add_white_noise(signal, noise_percentage):
    noise = np.random.randn(len(signal))
    noise_amp = noise_percentage * np.random.uniform() * np.amax(signal) / 100
    signal = signal + noise_amp * noise
    return signal


def pitch_shift(signal, steps):
    signal_tensor = torch.from_numpy(signal)
    pitch_shifter = T.PitchShift(22050, n_steps=steps)
    shifted_tensor = pitch_shifter(signal_tensor)
    return shifted_tensor.detach().numpy()


def pre_emphasis(signal, coef):
    signal_tensor = torch.from_numpy(signal)
    pre_emphasized_tensor = torch.cat((signal_tensor[0].unsqueeze(0), signal_tensor[1:] - coef * signal_tensor[:-1]))
    return pre_emphasized_tensor.numpy()


def dynamic_range_compression(signal, ratio):
    signal_tensor = torch.from_numpy(signal)
    rms = torch.sqrt(torch.mean(signal_tensor ** 2))
    compressed_audio_t = signal_tensor / torch.max(torch.abs(signal_tensor)) * rms * torch.Tensor([ratio])
    return compressed_audio_t.numpy()


def save_mfcc(dataset_path, json_path, augmentation, arg_2, num_mfcc, num_segments, hop_length):

    # sõnastik toonimistele
    augmentation_parameters = {
        "augmentation_name": augmentation,
        "augmentation_argument": arg_2,
        "num_mfcc": num_mfcc,
        "num_segments": num_segments,
        "hop_length": hop_length,
    }

    # peasõnastik
    data = {
        "augmentation_parameters": augmentation_parameters,
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "file_id": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            semantic_label = dirpath.split("/")[-1]
            print("semantic_label:", semantic_label)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            for f in filenames:
                print("f:", f, int(f.split(".")[1]))
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, dtype='float32')

                # toonimised
                if def_name in globals() and callable(globals()[def_name]):
                    def_func = globals()[def_name]
                    def_func(signal, def_arg_2)

                # segmentideks jagamised
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    y = signal[start:finish]

                    # librosa 0.10.0.post2
                    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=num_mfcc)

                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        data["file_id"].append(int(f.split(".")[1]))
                        print("{}, segment:{}".format(file_path, d + 1))

    # salvestame MFCC JSON faili
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    timer_start = int(time.time())

    for augmentation in augmentation_dicts:

        sub_timer_start = int(time.time())

        def_name = augmentation['def_name']
        def_arg_1 = augmentation['arg_1']
        def_arg_2 = augmentation['arg_2']

        if def_arg_2 == None:
            arg = ""
        else:
            arg = "_" + str(def_arg_2)
        JSON_FILE = str(def_name) + arg + "_" + str(timer_start) + ".json"

        # JSON folder
        if not os.path.exists(JSON_FOLDER):
            os.makedirs(JSON_FOLDER)

        JSON_PATH = JSON_FOLDER + JSON_FILE

        save_mfcc(DATASET_PATH, JSON_PATH,
                  augmentation=def_name,
                  arg_2=def_arg_2,
                  num_segments=NUM_SEGMENTS,
                  num_mfcc=NUM_MFCC,
                  hop_length=HOP_LENGTH)

        sub_timer_end = int(time.time())

    timer_end = int(time.time())
    print("augmentations process total time:", timer_end - timer_start)
