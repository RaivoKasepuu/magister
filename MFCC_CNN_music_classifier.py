# Muusika zhanri klassifitseerija
# Raivo Kasepuu
# B710710
# 13.04.2023

import time
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# Konstandid
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
N_FFT = 2048
HOP_LENGTH = 512
EPOCH = 200
BATCH_SIZE = 32
TEST_SIZE = 0.33
VALIDATION_SIZE = 0.25
LEARNING_RATE = 0.0001
FILE_MIN_SIZE = 1000000
RANDOM_STATE = 42


# Sisendandmestik
DATASETS_FOLDER = "/Users/raivo/Documents/Magister_files/augJSON"

# Väljund täpsuse ja kadude graafikud
PLOTS_FOLDER = DATASETS_FOLDER + "plots/"
if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

# Väljund segadusmaatriks
CM_PLOTS_FOLDER = DATASETS_FOLDER + "CM_plots/"
if not os.path.exists(CM_PLOTS_FOLDER):
    os.makedirs(CM_PLOTS_FOLDER)

# Väljund segadusmaatriks txt kujul
CM_TXTs_FOLDER = DATASETS_FOLDER + "CM_TXTs/"
if not os.path.exists(CM_TXTs_FOLDER):
    os.makedirs(CM_TXTs_FOLDER)

# Väljund JSON
RESULTS_JSON_FOLDER = DATASETS_FOLDER + "resJSONs/"
if not os.path.exists(RESULTS_JSON_FOLDER):
    os.makedirs(RESULTS_JSON_FOLDER)

# Väljund mudelite kaust
MODELS_FOLDER = DATASETS_FOLDER + "models/"
if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)


def get_datasets(data_path_original, data_path_augmented):
    """Loads training dataset from json files.
       :param data_path_original (str): Path to original json file
       :param data_path_augmented (str): Path to augmented json file
       :return data_original(ndarray): Original data dataset
       :return data_augmented(ndarray): Augmented data dataset
       :return original_model (boolean): original model setup boolean
       """
    with open(data_path_original, "r") as fp:
        data_original = json.load(fp)

    with open(data_path_augmented, "r") as fp:
        data_augmented = json.load(fp)

    # Originaali määramine toonimata andmestikuga
    original_model = False
    if DATASET_PATH_ORIGINAL == DATASET_PATH_AUGMENTED:
        original_model = True

    return data_original, data_augmented, original_model


def augmentation_id(augmentation_string, augmentation_argument):
    output_string = ""
    input_string_list = augmentation_string.split("_")
    for string in input_string_list:
        output_string += string[0]
    if augmentation_argument:
        if abs(augmentation_argument) < 1:
            arg_string = str(augmentation_argument).split(".")[-1]
        else:
            arg_string = str(augmentation_argument)
        output_string += arg_string
        output_string = output_string.capitalize()
    if augmentation_string[0: 4] == "orig":
        output_string = "ORIG"
    return output_string


def load_data(data_original, data_augmented, original_model):

    # Loome indeksid
    original_indices = np.arange(np.array(data_original["mfcc"]).shape[0])
    augmented_indices = np.arange(np.array(data_augmented["mfcc"]).shape[0])

    # Splitime test ja treeningandmestikud
    X_train_orig, X_test_orig, y_train_orig, y_test_orig, orig_indices_train, orig_indices_test = \
        train_test_split(np.array(data_original["mfcc"]), np.array(data_original["labels"]), original_indices,
                         test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train_aug, X_test_aug, y_train_aug, y_test_aug, aug_indices_train, aug_indices_test = \
        train_test_split(np.array(data_augmented["mfcc"]), np.array(data_augmented["labels"]), augmented_indices,
                         test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Lisame treeningandmestiku
    if original_model:
        X_train = X_train_orig
        y_train = y_train_orig
    else:
        X_train = np.concatenate((X_train_orig, X_train_aug))
        y_train = np.concatenate((y_train_orig, y_train_aug))

    # Testime ainult originaalandmetega
    X_test = X_test_orig
    y_test = y_test_orig

    # Kontrollime suurusi
    print("X_train shape:", X_train.shape[0])
    print("X_test shape:", X_test.shape[0])

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    # Lisame axise
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def plot_history(history, title, augmentation_sting):

    fig, axs = plt.subplots(2)

    # Täpsuse subplot
    title_acc = str(title) + "\n" + "Täpsus"
    axs[0].plot(history.history["accuracy"], label="treeningandmed")
    axs[0].plot(history.history["val_accuracy"], label="valideerimisandmed")
    axs[0].set_ylabel("Täpsus")
    axs[0].set_xlabel("Kordused")
    axs[0].legend(loc="lower right")
    axs[0].set_title(title_acc)

    # Kadude subplot
    title_loss = "Kadu"
    axs[1].plot(history.history["loss"], label="treeningandmed")
    axs[1].plot(history.history["val_loss"], label="valideerimisandmed")
    axs[1].set_ylabel("Kadu")
    axs[1].set_xlabel("Kordused")
    axs[1].legend(loc="upper right")
    axs[1].set_title(title_loss)

    # Graafikute vaheala
    fig.subplots_adjust(hspace=0.5)

    plt.show()
    # Salvestame faili
    plot_file = PLOTS_FOLDER + 'acc_loss_plot_' + str(augmentation_sting) + '.png'
    fig.savefig(plot_file)


def plot_confusion_matrix(confusion_matrix, classes, title, augmentation_string, normalize=True, cmap=plt.cm.Blues):
    """
    Segadusmaatriksi plot ja print
    Normaliseerimiseks pane `normalize=True`.
    """

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.subplots(figsize=(7, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Segadusmaatriks" + "\n" + title)
    # plt.colorbar(format='%.0f')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f%' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, '{:.0f}%'.format(confusion_matrix[i, j]*100),
                         horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel(' ')
    plt.xlabel(' ')
    CM_plot_filename = CM_PLOTS_FOLDER + augmentation_string + ".png"
    plt.savefig(CM_plot_filename)


def build_model(input_shape):
    """Ehitame CNN mudeli
    """

    # Topoloogia
    CNN_model = keras.Sequential()

    # Esimene kiht
    CNN_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    CNN_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    CNN_model.add(keras.layers.BatchNormalization())

    # Teine kiht
    CNN_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    CNN_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    #CNN_model.add(keras.layers.BatchNormalization())
    CNN_model.add(keras.layers.Dropout(0.3))

    # Kolmas kiht
    CNN_model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    CNN_model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    #CNN_model.add(keras.layers.BatchNormalization())
    CNN_model.add(keras.layers.Dropout(0.3))

    # Flatten kiht
    CNN_model.add(keras.layers.Flatten())
    CNN_model.add(keras.layers.Dense(64, activation='relu'))
    #CNN_model.add(keras.layers.Dropout(0.3))
    CNN_model.add(keras.layers.Dropout(0.3))

    # Väljundkiht
    CNN_model.add(keras.layers.Dense(10, activation='softmax'))

    return CNN_model


def predict(CNN_model, X, y):

    # Lisame dimensioonid - model.predict() eeldab 4D
    X = X[np.newaxis, ...]  # (1, 130, 13, 1)

    # Ennustus
    prediction = CNN_model.predict(X)

    # Max indeks
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":
    # Käime läbi JSON folder:
    augmented_files = []
    for filename in os.listdir(DATASETS_FOLDER):
        if filename.endswith(".json"):
            if os.path.getsize(DATASETS_FOLDER + filename) > FILE_MIN_SIZE:
                augmented_files.append(filename)
            if filename.split("_")[0] == "original":
                DATASET_PATH_ORIGINAL = DATASETS_FOLDER + filename
    # Failide sorteerimine
    augmented_files.sort(reverse=False)
    print(augmented_files)

    for aug_file in augmented_files:
        start_load = int(time.time())
        DATASET_PATH_AUGMENTED = DATASETS_FOLDER + aug_file
        augmentation_timestamp = aug_file.split("_")[-1].split(".")[0]

        # Andmestikkude laadimine
        data_original, data_augmented, original_model = get_datasets(DATASET_PATH_ORIGINAL, DATASET_PATH_AUGMENTED)
        print("DATASET_PATH_ORIGINAL:", DATASET_PATH_ORIGINAL)
        print("DATASET_PATH_AUGMENTED:", DATASET_PATH_AUGMENTED)
        genres_list = data_original["mapping"]
        parameters = data_augmented["augmentation_parameters"]
        augmentation = parameters["augmentation_name"]
        augmentation_argument = parameters["augmentation_argument"]
        NUM_MFCC = parameters["num_mfcc"]
        NUM_SEGMENTS = parameters["num_segments"]
        ID = augmentation_id(augmentation, augmentation_argument)

        conf_string = "[segments: " + str(NUM_SEGMENTS) + ", num_mfcc:" + str(NUM_MFCC) + "]"
        CONF = "___" + str(NUM_SEGMENTS) + "seg_" + str(NUM_MFCC) + "mfcc"

        if augmentation_argument:
            augmentation_string = augmentation + "_" + str(augmentation_argument)
        else:
            augmentation_string = augmentation
        print("augmentation_string", augmentation_string)
        title = "Mudeli ID: " + ID
        print("title:", title)

        # Treeningu, valideerimise ja testi jagamine
        X_train, X_validation, X_test, y_train, y_validation, y_test, = load_data(
            data_original,
            data_augmented,
            original_model
        )

        start = int(time.time())
        # Sisendi määramine
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        print("input_shape:", input_shape)

        model = build_model(input_shape)

        history = History()

        optimiser = keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimiser,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # Mudeli treenimine
        history = model.fit(X_train, y_train,
                            validation_data=(X_validation, y_validation),
                            batch_size=BATCH_SIZE,
                            epochs=EPOCH,
                            callbacks=[history])

        # Mudeli salvestamine
        model.save(os.path.join(MODELS_FOLDER, augmentation_string + "_model.h5"))

        # Plotime täpsuse ja kao treening ja valideerimisandmestikega
        plot_history(history, title, augmentation_string)

        # Teeme testi
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print("test_ID:", augmentation_string, "Test accuracy:", test_acc)

        # Ennustame
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Loome segadusmaatriksi
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=genres_list, title=title,
                              augmentation_string=augmentation_string)
        plt.show()

        # Prindime segadusmaatriksi
        print("Segadusmaatriks:")
        print(cm)
        print()

        # Prindime normaliseeritud segadusmaatriksi
        print("Normaliseeritud segadusmaatriks:")
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cmn)
        print()

        # Konverteerime segadusmaatriksid sõnastikkudesse
        conf_matrix_str = cm.tolist()
        cm_txt_file = CM_TXTs_FOLDER + augmentation_string + ".txt"

        norm_conf_matrix_str = cmn.tolist()
        norm_cm_txt_file = CM_TXTs_FOLDER + "norm_" + augmentation_string + ".txt"

        # Salvestamine segadusmaatriksi
        with open(cm_txt_file, 'w') as f:
            json.dump(conf_matrix_str, f)

        # Salvestame normaliseeeritud segadusmaatriksi
        with open(norm_cm_txt_file, 'w') as f:
                json.dump(norm_conf_matrix_str, f)

        end = int(time.time())
        model_training_time = end - start
        print("model training time: ", model_training_time)

        # Testennustuse valimine
        X_to_predict = X_test[1]
        y_to_predict = y_test[1]

        # Ennustame testennustuse
        predict(model, X_to_predict, y_to_predict)

        # Tulemuste sõnastik:
        results_dict = {'augmentation': augmentation, 'augmentation_argument': augmentation_argument,
                        'augmentation_timestamp': augmentation_timestamp, 'augmentation_string': augmentation_string,
                        'ID': ID, 'test_accuracy': test_acc, 'epochs': EPOCH,  'learning_rate': LEARNING_RATE,
                        'batch_size': BATCH_SIZE, 'test_ratio': TEST_SIZE, 'validation_size': VALIDATION_SIZE,
                        'num_segments': NUM_SEGMENTS, 'num_MFCC': NUM_MFCC,
                        'start_load': int(start_load), 'start_time': int(start),
                        'end_time': int(end), 'model_training_time': int(model_training_time)}
        print(results_dict)
        print()

        # Salvestame tulemused JSON faili:
        results_json_filename = RESULTS_JSON_FOLDER + augmentation_string + "_" + str(start) + "_result.json"

        # JSON kausta kontroll ja loomine
        if not os.path.exists(RESULTS_JSON_FOLDER):
            # loome kausta
            os.makedirs(RESULTS_JSON_FOLDER)

        print("results_json_filename", results_json_filename)
        with open(results_json_filename, 'w') as f:
            json.dump(results_dict, f)





