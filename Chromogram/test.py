import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Constants
G1_FREQ = 48.9994  # Frequency of G1
FS = 44100  # Sampling rate
DURATION = 2.0  # Duration in seconds
N_FFT = 2048  # Number of FFT points
HOP_LENGTH = 512  # Hop length
WINDOW = 'hann'  # Window function
GAMMA = 1.0  # Compression parameter for logarithmic compression
NROOT = 5  # Root parameter for root compression

# Note labels for 6 octaves starting from G1
note_labels = [
    'G1', 'G#1', 'A1', 'A#1', 'B1', 'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2',
    'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3',
    'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4',
    'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5',
    'G5', 'G#5', 'A5', 'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6',
    'G6', 'G#6', 'A6', 'A#6', 'B6', 'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7'
]

def generate_sinewave(frequency, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * frequency * t)

def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=FS, mono=False)
    if y.ndim > 1:
        y = np.mean(y, axis=0)  # Convert to mono by averaging channels
    return y, sr

def compute_stft(signal, n_fft, hop_length, window):
    return np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window))

def pitch_to_bin(frequency, g1_freq):
    # Avoid log of zero or negative frequencies
    if frequency > 0:
        return 12 * np.log2(frequency / g1_freq)
    else:
        return -np.inf  # Indicate invalid frequency

def compute_pitch_energies(S, fs, g1_freq):
    freqs = np.linspace(0, fs // 2, S.shape[0])
    pitch_bins = np.array([pitch_to_bin(f, g1_freq) for f in freqs])
    pitch_energies = np.zeros((72, S.shape[1]))  # 72 pitch bins (6 octaves)

    for i in range(S.shape[0]):
        if pitch_bins[i] != -np.inf:
            pitch_class = int(np.round(pitch_bins[i])) % 12  # Map to 12 pitch classes starting from G
            octave = int(np.floor(pitch_bins[i] / 12))
            if 0 <= pitch_class < 12 and 0 <= octave < 6:  # Only consider the defined range
                pitch_energies[pitch_class + 12 * octave, :] += S[i, :]

    return pitch_energies

def fold_into_chromagram(pitch_energies):
    chromagram = np.zeros((72, pitch_energies.shape[1]))
    for i in range(6):
        for j in range(12):
            if i % 2 == 0:
                chromagram[j, :] += pitch_energies[j + 12 * i, :]  # Sum for even octaves
            else:
                chromagram[j + 12, :] += pitch_energies[j + 12 * i, :]  # Sum for odd octaves
    return chromagram

def apply_compression(energies, method='log', gamma=1.0, nroot=5):
    if method == 'log':
        return np.log(1 + gamma * energies)
    elif method == 'root':
        return energies ** (1 / nroot)
    else:  # linear
        return energies

# Generate test sinewaves
frequencies = [65.4, 130.8, 261.6]  # C2, C3, C4
sinewaves = [generate_sinewave(f, DURATION, FS) for f in frequencies]
combined_signal = np.sum(sinewaves, axis=0)

# Or load an audio file
# combined_signal, FS = load_audio("path_to_audio_file.mp3")

# Compute STFT
Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, WINDOW)

# Compute pitch energies
pitch_energies = compute_pitch_energies(Stft, FS, G1_FREQ)

# Fold into chromagram (covering 6 octaves starting from G1)
chromagram = fold_into_chromagram(pitch_energies)

# Apply compression
compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

# Plotting the signal, spectrogram, and pitch energies
fig1, axs1 = plt.subplots(3, 1, figsize=(12, 12))

# a. Signal amplitude vs time
axs1[0].plot(np.linspace(0, DURATION, len(combined_signal)), combined_signal)
axs1[0].set_title('Signal Amplitude vs Time')
axs1[0].set_xlabel('Time (s)')
axs1[0].set_ylabel('Amplitude')

# b. Spectrogram
img1 = librosa.display.specshow(librosa.amplitude_to_db(Stft, ref=np.max), sr=FS, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', ax=axs1[1])
fig1.colorbar(img1, format='%+2.0f dB', ax=axs1[1])
axs1[1].set_title('Spectrogram')

# c. Pitch Energies before folding into chromagram
img2 = axs1[2].imshow(pitch_energies, aspect='auto', origin='lower', interpolation='none')
fig1.colorbar(img2, ax=axs1[2])
axs1[2].set_title('Pitch Energies before folding into Chromagram')
axs1[2].set_xlabel('Time')
axs1[2].set_ylabel('Pitch Bins')

plt.tight_layout()

# Plotting the compressed chromagram in a separate window
fig2, ax2 = plt.subplots(figsize=(12, 12))

img3 = librosa.display.specshow(compressed_chromagram, x_axis='time', sr=FS, hop_length=HOP_LENGTH, y_axis=None, cmap='coolwarm', ax=ax2)
ax2.set_yticks(np.arange(72))
ax2.set_yticklabels(note_labels)
fig2.colorbar(img3, format='%+2.0f dB', ax=ax2)
plt.title('Compressed Chromagram (6 Octaves from G1)')
plt.tight_layout()

plt.show()
