import librosa
import numpy as np
import matplotlib.pyplot as plt
import random

#Show given an mp3 filepath
def show_spectrogram_filepath(file_path):
    ex, sr = librosa.load(file_path, sr=None)

    show_spectrogram(ex)

# Show based on just the raw audio data
def show_spectrogram_audiodata(ex):
    S = np.abs(librosa.stft(ex))

    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    show_spectrogram(S_dB)


#Show given the array already in dB form
def show_spectrogram(S_dB):
    fig, ax = plt.subplots()

    img = librosa.display.specshow(S_dB, y_axis='log', x_axis='time', ax=ax)

    ax.set_title('Power Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")


#Show the difference between 2 mp3s.
#They might have different max and min values so I normalize after getting the difference
#Normalizing is done by taking the difference and dividing by the range of the values between the two arrays to get a percentage difference
def diff_mp3(array, other):

    popMax = max(np.max(array), np.max(other))
    popMin = min(np.min(array), np.min(other))

    total_diff = np.abs(array - other)

    total_diff = total_diff / (popMax - popMin)

    print("Average:", np.average(total_diff))
    print("Stdev:", np.std(total_diff))
    print("Max:", np.max(total_diff))
    print("Min:", np.min(total_diff))


#Meant to be used with dB values
#Mess percent is positive and negative direction, so 10% would be +- 10%
def mess_values(input, mess_percent):
    fmax = np.max(input)
    fmin = np.min(input)

    new_arr = np.copy(input)

    range = fmax - fmin

    diff = range * mess_percent / 50

    for y in new_arr:
        for x in y:
            x += (random.uniform(0, 1) * diff) - (diff / 2)

    return new_arr

