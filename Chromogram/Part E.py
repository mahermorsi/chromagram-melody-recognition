from Chromagram import *
from scipy import signal
def main():

    signal_2, fs = librosa.load("Twinkle Harmony.wav")
    signal_1, fs = librosa.load("Twinkle melody.wav")

    #define parameters
    N_FFT = 2**14
    HOP_LENGTH = 512
    WIN_LENGTH = int(N_FFT/8)
    GAMMA = 1

    # Compute STFT
    Stft_1 = compute_stft(signal_1, N_FFT, HOP_LENGTH, signal.windows.hann(WIN_LENGTH), WIN_LENGTH)
    Stft_2 = compute_stft(signal_2, N_FFT, HOP_LENGTH, signal.windows.hann(WIN_LENGTH), WIN_LENGTH)


    # Compute pitch energies
    pitch_energies_1 = compute_pitch_energies(Stft_1, fs, G1_FREQ)
    pitch_energies_2 = compute_pitch_energies(Stft_2, fs, G1_FREQ)

    #compressing the pitch energies
    #compressed_pitch = apply_compression(pitch_energies, method='log', gamma=GAMMA)

    #compute chroma-gram of two octaves
    chromagram_1 = fold_into_chromagram(pitch_energies_1)
    chromagram_2 = fold_into_chromagram(pitch_energies_2)

    # Apply compression
    compressed_chromagram_1 = apply_compression(chromagram_1, method='log', gamma=GAMMA)
    compressed_chromagram_2 = apply_compression(chromagram_2, method='log', gamma=GAMMA)

    # Plotting
    plot_compressed_chromagram(compressed_chromagram_1, chroma_labels, fs, HOP_LENGTH, WIN_LENGTH )
    plot_compressed_chromagram(compressed_chromagram_2, chroma_labels, fs, HOP_LENGTH, WIN_LENGTH )


    # Show all plots at once
    plt.show()


if __name__ == "__main__":
    main()