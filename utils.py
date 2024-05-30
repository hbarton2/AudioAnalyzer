import librosa
import numpy as np
import matplotlib.pyplot as plt

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


def diff_mp3(array, other):
    total_diff = np.sum(np.abs(array - other))

    print("Average:", np.average(total_diff))
    print("Stdev:", np.std(total_diff))