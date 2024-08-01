import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Constants
G1_FREQ = 48.9994  # Frequency of G1
FS = 44100  # Sampling rate
DURATION = 2.0  # Duration in seconds
N_FFT = 2**14  # Number of FFT points
HOP_LENGTH = 512  # Hop length
WINDOW = 'hann'  # Window function
GAMMA = 0.002  # Compression parameter for logarithmic compression
NROOT = 5  # Root parameter for root compression
WIN_LENGTH = int(N_FFT/2)

# Window overlap = (N_FFT - HOP_LENGTH) / N_FFT
# For this configuration: overlap = (16384 - 512) / 16384 = 0.96875 or 96.875%

# Chromagram labels for odd and even octaves
chroma_labels = [
    'G_odd', 'G#_odd', 'A_odd', 'A#_odd', 'B_odd', 'C_odd', 'C#_odd', 'D_odd', 'D#_odd', 'E_odd', 'F_odd', 'F#_odd',
    'G_even', 'G#_even', 'A_even', 'A#_even', 'B_even', 'C_even', 'C#_even', 'D_even', 'D#_even', 'E_even', 'F_even', 'F#_even'
]

# Note labels for 6 octaves starting from G1
note_labels = [
    'G1', 'G#1', 'A1', 'A#1', 'B1', 'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2',
    'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3',
    'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4',
    'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5',
    'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6',
    'G6', 'G#6', 'A6', 'A#6', 'B6', 'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7'
]

# Generate a sinewave for given frequency, duration, and sampling rate
def generate_sinewave(frequency, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

# Load audio file, converting to mono if necessary
def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=FS, mono=False)
    if y.ndim > 1:
        y = np.mean(y, axis=0)  # Convert to mono by averaging channels
    return y, sr

# Compute the Short-Time Fourier Transform (STFT) of the signal
def compute_stft(signal, n_fft, hop_length, window, winLength):
    return np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window, win_length=winLength))

# Convert pitch index to frequency
def pitch_to_frequency(pitch_index, g1_freq):
    return g1_freq * (2 ** (pitch_index / 12))

# Compute pitch energies for each pitch bin in the STFT matrix
def compute_pitch_energies(S, fs, g1_freq):
    freqs = np.linspace(0, fs // 2, S.shape[0])
    num_pitches = 72  # 6 octaves * 12 pitches per octave
    pitch_energies = np.zeros((num_pitches, S.shape[1]))  # 72 pitch bins (6 octaves)

    for pitch_index in range(num_pitches):
        pitch_freq = pitch_to_frequency(pitch_index, g1_freq)
        lower_bound = pitch_freq * (2 ** (-0.5 / 12))
        upper_bound = pitch_freq * (2 ** (0.5 / 12))
        lower_bin = np.searchsorted(freqs, lower_bound)
        upper_bin = np.searchsorted(freqs, upper_bound)
        if lower_bin < upper_bin:
            pitch_energies[pitch_index, :] = np.sum(S[lower_bin:upper_bin, :], axis=0)
        else:
            pitch_energies[pitch_index, :] = S[lower_bin, :]

    return pitch_energies

# Fold pitch energies into a chromagram, summing odd and even octaves separately
def fold_into_chromagram(pitch_energies):
    chromagram = np.zeros((24, pitch_energies.shape[1]))
    for i in range(6):  # Fold the 6 octaves into 24 pitch classes
        for j in range(12):
            if i % 2 == 0:
                chromagram[j, :] += pitch_energies[j + 12 * i, :]  # Sum for even octaves
            else:
                chromagram[j + 12, :] += pitch_energies[j + 12 * i, :]  # Sum for odd octaves
    return chromagram

# Apply compression to the chromagram
def apply_compression(energies, method='log', gamma=1.0, nroot=5):
    if method == 'log':
        return np.log2(1 + gamma * energies)
    elif method == 'root':
        return energies ** (1 / nroot)
    else:  # linear
        return energies

# Plotting functions
def plot_signal_and_spectrogram(signal, stft, duration, fs, hop_length, winLength):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # a. Signal amplitude vs time
    axs[0].plot(np.linspace(0, duration, len(signal)), signal)
    axs[0].set_title('Signal Amplitude vs Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')

    # b. Spectrogram
    img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[1], win_length=winLength)
    fig.colorbar(img, format='%+2.0f dB', ax=axs[1])
    axs[1].set_title('Spectrogram')

    plt.tight_layout()

def plot_pitch_energies(pitch_energies, note_labels, cmap):
    fig, ax = plt.subplots(figsize=(12, 12))

    img = ax.imshow(pitch_energies, aspect='auto', origin='lower', interpolation='none', cmap=cmap)
    fig.colorbar(img, ax=ax)
    ax.set_title('Pitch Energies before folding into Chromagram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pitch Bins')
    ax.set_yticks(np.arange(len(note_labels)))
    ax.set_yticklabels(note_labels)

    plt.tight_layout()

def plot_compressed_chromagram(chromagram, chroma_labels, fs, hop_length):
    fig, ax = plt.subplots(figsize=(12, 12))

    img = librosa.display.specshow(chromagram, x_axis='time', sr=fs, hop_length=hop_length, y_axis=None, cmap='coolwarm', ax=ax)
    ax.set_yticks(np.arange(24))
    ax.set_yticklabels(chroma_labels)  # Only show two octaves worth of labels
    fig.colorbar(img, format='%+2.0f dB', ax=ax)
    plt.title('Compressed Chromagram (2 Octaves)')
    plt.tight_layout()

# Main code
