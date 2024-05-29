import librosa
import numpy as np
import matplotlib.pyplot as plt

def show_spectrogram_filepath(file_path):
    ex, sr = librosa.load(file_path, sr=None)

    show_spectrogram(ex)



def show_spectrogram(ex):
    S = np.abs(librosa.stft(ex))

    fig, ax = plt.subplots()

    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)

    ax.set_title('Power Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")