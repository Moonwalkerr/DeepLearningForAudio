# importing required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# loading the audio file into an array
piano_file = "../audios/piano_c.wav"
piano, sr = librosa.load(piano_file)

# plotting the waveplot
def plot_waveplot(signal,sr):
    librosa.display.waveplot(signal,sr=sr)
    plt.title("Waveplot")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("waveplot")

# fft feature extraction and plotting
def plot_magnitude_spectrum(signal, sr):
    # extracting fft
    signal_fft = np.fft.fft(signal)
    magnitude_signal = np.abs(signal_fft)
    freq_bins = np.linspace(0,sr,len(magnitude_signal))
    # freq bins below nyquist freq 
    left_freq_bins = freq_bins[:int(len(freq_bins)/2)]
    left_magnitude = magnitude_signal[:int(len(magnitude_signal)/2)]
    # plotting the magnitude spectrum
    plt.plot(left_freq_bins,left_magnitude,color='red')
    plt.title("Power / Magnitude spectrum of audio signal")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.savefig("Magnitude_Spec_signal")
    print("Done !")


# extracting and plotting the stfts / spectogram
def plot_spectogram(signal,sr,frame_length=2048,hop_length=512):
    signal_stft = librosa.core.stft(signal,n_fft=frame_length,hop_length=hop_length)
    spectogram = np.abs(signal_stft)**2
    # this spectogram is currently linear
    # humans perceive the spectogram in logarithmic way
    # making our spectogram perceivable to human --> decibles
    spec_log = librosa.power_to_db(spectogram)

    # plotting the spectogram
    plt.figure(figsize=(10,10))
    librosa.display.specshow(spec_log,sr=sr,hop_length=hop_length,x_axis='time',y_axis='log')
    plt.colorbar()
    plt.title("Spectogram of the Signal")
    plt.savefig("Spectogram")
    

FRAME_LENGTH = 2048
HOP_LENGTH = 512

# plot_spectogram(signal=piano,sr=sr,frame_length=FRAME_LENGTH,hop_length=HOP_LENGTH)


#  Extraction and plotting of mfccs
def plot_mfccs(signal,sr,frame_length=2028,hop_length=512,n_mels=13):
    mfcc_signal = librosa.feature.mfcc(signal,sr,n_mels=n_mels,n_fft=frame_length,hop_length=hop_length)
    # power to db
    mfcc_signal_log = librosa.power_to_db(mfcc_signal)

    # plotting the mfcc
    librosa.display.specshow(mfcc_signal_log,x_axis='time',y_axis='mel')
    plt.colorbar()
    plt.title("MFCC of the audio signal")
    plt.xlabel("Time")
    plt.ylabel("Mel frequency Coefficents (MFCCs)")
    plt.savefig("MFCCsignal")

plot_mfccs(piano,sr=sr,frame_length=FRAME_LENGTH,hop_length=HOP_LENGTH)