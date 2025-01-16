import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Your dynamic EQ plugin class
class DynamicEQPlugin:
    def __init__(self):
        self.sample_rate = 44100
        self.bands = [
            {'band': (20, 200), 'threshold_db': 0, 'ratio': 2, 'makeup_gain': 0},    # Low frequencies
            {'band': (200, 2000), 'threshold_db': 0, 'ratio': 2, 'makeup_gain': 0},  # Mid frequencies
            {'band': (2000, 20000), 'threshold_db': 0, 'ratio': 2, 'makeup_gain': 0}  # High frequencies
        ]

    def process_audio(self, audio):
        magnitude, phase = self.get_spectrum(audio)
        processed_magnitude = self.multiband_dynamic_eq(magnitude)
        smoothed_magnitude = self.smooth_magnitude(processed_magnitude)
        processed_audio = self.reconstruct_audio(smoothed_magnitude, phase)
        return processed_audio

    def get_spectrum(self, audio):
        n_fft = 2048
        hop_length = 512
        magnitude, phase = [], []

        for channel in range(audio.shape[0]):
            D = librosa.stft(audio[channel], n_fft=n_fft, hop_length=hop_length)
            mag, phs = librosa.magphase(D)
            magnitude.append(mag)
            phase.append(phs)

        return np.array(magnitude), np.array(phase)

    def multiband_dynamic_eq(self, magnitude):
        compressed_magnitude = np.copy(magnitude)

        for band_info in self.bands:
            band = band_info['band']
            threshold_db = band_info['threshold_db']
            ratio = band_info['ratio']
            makeup_gain = band_info['makeup_gain']

            band_magnitude = self.extract_band(magnitude, band)
            compressed_band_magnitude, calculated_gain = self.dynamic_eq(band_magnitude, threshold_db, ratio)

            if makeup_gain == 'auto':
                # Clamp makeup gain to avoid excessive boosts
                makeup_gain = min(max(calculated_gain, 0.8), 1.2)

            compressed_band_magnitude *= makeup_gain
            compressed_magnitude = self.replace_band(compressed_magnitude, band, compressed_band_magnitude)

        return compressed_magnitude

    def extract_band(self, magnitude, band):
        freq_bin_start = int(band[0] * magnitude.shape[1] / self.sample_rate)
        freq_bin_end = int(band[1] * magnitude.shape[1] / self.sample_rate)
        return magnitude[:, freq_bin_start:freq_bin_end, :]

    def replace_band(self, compressed_magnitude, band, compressed_band_magnitude):
        freq_bin_start = int(band[0] * compressed_magnitude.shape[1] / self.sample_rate)
        freq_bin_end = int(band[1] * compressed_magnitude.shape[1] / self.sample_rate)
        compressed_magnitude[:, freq_bin_start:freq_bin_end, :] = compressed_band_magnitude
        return compressed_magnitude

    def dynamic_eq(self, magnitude, threshold_db, ratio):
        threshold = librosa.db_to_amplitude(threshold_db)
        compressed_magnitude = np.copy(magnitude)

        total_gain_reduction = 0
        non_zero_elements = 0

        for i in range(compressed_magnitude.shape[1]):
            for j in range(compressed_magnitude.shape[2]):
                for channel in range(compressed_magnitude.shape[0]):
                    original_value = compressed_magnitude[channel, i, j]
                    if original_value > threshold:
                        compressed_magnitude[channel, i, j] = threshold + ((original_value - threshold) / ratio)
                        gain_reduction = original_value - compressed_magnitude[channel, i, j]
                        total_gain_reduction += gain_reduction
                        non_zero_elements += 1

        average_gain_reduction = (total_gain_reduction / non_zero_elements) if non_zero_elements > 0 else 0
        auto_makeup_gain = 1.0 + (average_gain_reduction / 10) if average_gain_reduction > 0 else 1.0

        return compressed_magnitude, auto_makeup_gain

    def smooth_magnitude(self, magnitude):
        # Apply smoothing to the magnitude spectrum to reduce artifacts
        smoothed_magnitude = np.copy(magnitude)
        for channel in range(smoothed_magnitude.shape[0]):
            for i in range(1, smoothed_magnitude.shape[1] - 1):
                smoothed_magnitude[channel, i, :] = (
                    smoothed_magnitude[channel, i - 1, :] +
                    smoothed_magnitude[channel, i, :] +
                    smoothed_magnitude[channel, i + 1, :]
                ) / 3
        return smoothed_magnitude

    def reconstruct_audio(self, compressed_magnitude, phase):
        D_compressed = compressed_magnitude * phase
        processed_audio = np.array([librosa.istft(D_compressed[channel]) for channel in range(D_compressed.shape[0])])
        return processed_audio


# Plot Frequency Response (EQ Curves)
def plot_combined_frequency_response(original_signal, processed_signal, sample_rate):
    # Compute FFT for both signals
    def compute_frequency_response(signal):
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
        positive_freqs = freqs[:len(freqs) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]
        dB = 20 * np.log10(np.maximum(positive_magnitude, 1e-10))  # Avoid log(0)
        return positive_freqs, dB

    original_freqs, original_dB = compute_frequency_response(original_signal)
    processed_freqs, processed_dB = compute_frequency_response(processed_signal)

    # Plot original and processed frequency responses
    plt.figure(figsize=(12, 6))

    # Original Signal
    plt.subplot(1, 2, 1)  # Subplot (1 row, 2 columns, first plot)
    plt.plot(original_freqs, original_dB, label="Original", color='blue', alpha=0.7)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Original Frequency Response")
    plt.grid(True)

    # Processed Signal
    plt.subplot(1, 2, 2)  # Subplot (1 row, 2 columns, second plot)
    plt.plot(processed_freqs, processed_dB, label="Processed", color='red', alpha=0.7)
    plt.xscale('log')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Processed Frequency Response")
    plt.grid(True)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def generate_test_waveform(sample_rate=44100, duration=2.0):
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Define fundamental frequency and harmonics
    fundamental_freq = 440  # A4 (440 Hz)
    harmonics = [fundamental_freq * i for i in range(1, 6)]  # Harmonics: 1x, 2x, 3x, ..., 5x

    # Create a signal with fundamental and harmonics
    signal = np.sin(2 * np.pi * harmonics[0] * t)  # Fundamental
    for harmonic in harmonics[1:]:
        signal += (1 / harmonic) * np.sin(2 * np.pi * harmonic * t)  # Add harmonics with decreasing amplitude

    # Add some high-frequency noise to simulate harshness
    signal += 0.2 * np.sin(2 * np.pi * 8000 * t)  # 8 kHz noise

    # Normalize the signal to prevent clipping
    signal /= np.max(np.abs(signal))

    return signal

# Instantiate and test the plugin
plugin = DynamicEQPlugin()

# Generate and process the test waveform
sample_rate = 44100
test_signal = generate_test_waveform(sample_rate)
processed_signal = plugin.process_audio(np.expand_dims(test_signal, axis=0))[0]

# Plot both pre- and post-EQ frequency responses
plot_combined_frequency_response(test_signal, processed_signal, sample_rate)