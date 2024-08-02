from Chromagram import *
G1_FREQ = 48.9994
Fsharp_FREQ = 92.50
C4_FREQ = 261.63
NROOT = 4
FS = 44100  # Sampling rate
DURATION = 2.0  # Duration in seconds
N_FFT = 2**14  # Number of FFT points
HOP_LENGTH = 512  # Hop length
WINDOW = 'hann'  # Window function
GAMMA = 0.002  # Compression parameter for logarithmic compression
WIN_LENGTH = int(N_FFT/2)
def main():
    WIN_LENGTH = int(N_FFT)
    frequencies = [G1_FREQ, Fsharp_FREQ, C4_FREQ]
    for freq in frequencies:
        # generate sinewave
        frequency = [freq]
        sinewaves = [generate_sinewave(f, DURATION, FS) for f in frequency]
        combined_signal = np.sum(sinewaves, axis=0)

        # Compute STFT
        Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, WINDOW, WIN_LENGTH)

        # Compute pitch energies
        pitch_energies = compute_pitch_energies(Stft, FS, G1_FREQ)
        if freq == G1_FREQ:
            compressed_energies = apply_compression(pitch_energies, method='log', gamma=GAMMA)
            compressed_energies = apply_compression(compressed_energies, method='root', nroot=NROOT)
            chromagram = fold_into_chromagram(compressed_energies)

        else:
            chromagram = fold_into_chromagram(pitch_energies)


        # Apply compression


        # Plotting
        plot_signal_and_spectrogram(combined_signal, Stft, DURATION, FS, HOP_LENGTH, WIN_LENGTH)
        plot_pitch_energies(pitch_energies, note_labels, cmap='inferno')
        plot_compressed_chromagram(chromagram, chroma_labels, FS, HOP_LENGTH)

        # Show all plots at once
        plt.show()

if __name__ == "__main__":
    main()
