from Chromagram import *

def main():

    combined_signal, fs = librosa.load("G5 slow fast.wav")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, WINDOW)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, fs, G1_FREQ)

    #compute chroma-gram of two octaves
    chromagram = fold_into_chromagram(pitch_energies)

    # Apply compression
    compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, fs, HOP_LENGTH)
    plot_pitch_energies(pitch_energies, note_labels)
    plot_compressed_chromagram(compressed_chromagram, chroma_labels, fs, HOP_LENGTH)

    # Show all plots at once
    plt.show()


if __name__ == "__main__":
    main()
