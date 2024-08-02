from Chromagram import *
from scipy import signal


def main():
    GAMMA = 0.01
    N_FFT = 2 ** 14
    WIN_LENGTH = int(N_FFT / 2)
    FS = 44100
    WINDOW = 'hann'

    combined_signal, fs = librosa.load("G1 3 5.wav")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, signal.windows.hamming(WIN_LENGTH), WIN_LENGTH)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, fs, G1_FREQ)
    compressed_energies = apply_compression(pitch_energies, method='log', gamma=GAMMA)

    #compute chroma-gram of two octaves
    chromagram = fold_into_chromagram(compressed_energies)

    # Apply compression


    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, fs, HOP_LENGTH, WIN_LENGTH)
    plot_pitch_energies(compressed_energies, note_labels, 'inferno')
    plot_compressed_chromagram(chromagram, chroma_labels, fs, HOP_LENGTH)

    # Show all plots at once
    plt.show()


if __name__ == "__main__":
    main()
