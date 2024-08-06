from Chromagram import *
from scipy import signal
def main():

    #define parameters
    N_FFT = 2**14
    WIN_LENGTH = int(N_FFT/4)
    HOP_LENGTH = int(WIN_LENGTH/16)
    GAMMA = 2

#PART A
    combined_signal, fs = librosa.load("A4 clarinet.wav")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, signal.windows.hamming(WIN_LENGTH), WIN_LENGTH)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, fs, G1_FREQ)
    #compressing the pitch energies
    compresed_pitch = apply_compression(pitch_energies, method='log', gamma=GAMMA)

    #compute chroma-gram of two octaves
    chromagram = fold_into_chromagram(pitch_energies)

    # Apply compression
    compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, fs, HOP_LENGTH,WIN_LENGTH)
    plot_pitch_energies(compresed_pitch, note_labels)
    #plot_compressed_chromagram(compressed_chromagram, chroma_labels, fs, HOP_LENGTH, WIN_LENGTH )

    # Show all plots at once
    plt.show()

#PART B
    combined_signal, fs = librosa.load("Flute Cymbol.wav")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, signal.windows.hamming(WIN_LENGTH), WIN_LENGTH)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, fs, G1_FREQ)
    #compressing the pitch energies
    compresed_pitch = apply_compression(pitch_energies, method='log', gamma=GAMMA)

    #compute chroma-gram of two octaves
    chromagram = fold_into_chromagram(pitch_energies)

    # Apply compression
    compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, fs, HOP_LENGTH,WIN_LENGTH)
    plot_pitch_energies(compresed_pitch, note_labels)
    #plot_compressed_chromagram(compressed_chromagram, chroma_labels, fs, HOP_LENGTH, WIN_LENGTH )

    # Show all plots at once
    plt.show()

#PART C
    combined_signal, fs = librosa.load("G3 piano cym.wav")

    # Compute STFT
    Stft = compute_stft(combined_signal, N_FFT, HOP_LENGTH, signal.windows.hamming(WIN_LENGTH), WIN_LENGTH)

    # Compute pitch energies
    pitch_energies = compute_pitch_energies(Stft, fs, G1_FREQ)
    #compressing the pitch energies
    compresed_pitch = apply_compression(pitch_energies, method='log', gamma=GAMMA)

    #compute chroma-gram of two octaves
    chromagram = fold_into_chromagram(pitch_energies)

    # Apply compression
    #compressed_chromagram = apply_compression(chromagram, method='log', gamma=GAMMA)

    # Plotting
    plot_signal_and_spectrogram(combined_signal, Stft, DURATION, fs, HOP_LENGTH,WIN_LENGTH)
    plot_pitch_energies(compresed_pitch, note_labels)
    #plot_compressed_chromagram(compressed_chromagram, chroma_labels, fs, HOP_LENGTH, WIN_LENGTH )

    # Show all plots at once
    plt.show()


if __name__ == "__main__":
    main()