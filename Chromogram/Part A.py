from Chromagram import *

def main():
    WIN_LENGTH = int(N_FFT / 2)
    # Generate test sinewaves for G2 and C4
    frequencies = [261.63]  # G2, C4
    sinewaves = [generate_sinewave(f, DURATION, FS) for f in frequencies]
    combined_signal = np.sum(sinewaves, axis=0)

    # Or load an audio file
    # combined_signal, FS = load_audio("path_to_audio_file.mp3")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, WINDOW, WIN_LENGTH)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, FS, G1_FREQ)

    # Fold into chromagram (covering 2 octaves starting from G1)
    chromagram = fold_into_chromagram(pitch_energies)

    # Apply compression
    compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, FS, HOP_LENGTH, WIN_LENGTH)
    plot_pitch_energies(pitch_energies, note_labels, cmap='inferno')
    plot_compressed_chromagram(compressed_chromagram, chroma_labels, FS, HOP_LENGTH)

    # Show all plots at once
    plt.show()

if __name__ == "__main__":
    main()
