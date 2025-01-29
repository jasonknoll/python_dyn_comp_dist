import os
import numpy as np
import librosa
import soundfile as sf

class DynamicEQPlugin:
    def __init__(self):
        self.sample_rate = 44100  # Default sample rate
        
        # Frequency bands: [Low, Mid, High]
        self.bands = [
            {'band': (20, 400), 'threshold_db': 0, 'ratio': 1, 'makeup_gain': 1},    # Low frequencies
            {'band': (400, 8000), 'threshold_db': -12, 'ratio': 2, 'makeup_gain': 'auto'},  # Mid frequencies
            {'band': (8000, 20000), 'threshold_db': 0, 'ratio': 1, 'makeup_gain': 1}  # High frequencies
        ]
        self.fundamental_tolerance = 1  # Hz, tolerance for excluding frequencies
        self.max_harmonics = 8  # Number of harmonics to exclude
        self.noise_threshold = -30

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
        return audio, sr

    def process_audio(self, audio, fundamentals, include_fundamentals=False):
        magnitude, phase = self.get_spectrum(audio)
        reconstructed_magnitude = self.multiband_dynamic_eq(magnitude, fundamentals, include_fundamentals)
        processed_audio = self.reconstruct_audio(reconstructed_magnitude, phase)
        return processed_audio

    def get_spectrum(self, audio):
        n_fft = 4096
        hop_length = n_fft // 4
        magnitude, phase = [], []

        for channel in range(audio.shape[0]):
            D = librosa.stft(audio[channel], n_fft=n_fft, hop_length=hop_length)
            mag, phs = librosa.magphase(D)
            magnitude.append(mag)
            phase.append(phs)
        
        return np.array(magnitude), np.array(phase)

    def multiband_dynamic_eq(self, magnitude, fundamentals, include_fundamentals):
        compressed_magnitude = np.copy(magnitude)

        for band_info in self.bands:
            band = band_info['band']
            threshold_db = band_info['threshold_db']
            ratio = band_info['ratio']
            makeup_gain = band_info['makeup_gain']

            band_magnitude = self.extract_band(magnitude, band)
            compressed_band_magnitude = self.dynamic_eq(band_magnitude, threshold_db, ratio, fundamentals, include_fundamentals)

            if makeup_gain == 'auto':
                makeup_gain = self.calculate_auto_makeup_gain(band_magnitude, compressed_band_magnitude)
            compressed_band_magnitude *= makeup_gain

            compressed_band_magnitude = self.smooth_magnitude(compressed_band_magnitude)
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

    def dynamic_eq(self, magnitude, threshold_db, ratio, fundamentals, include_fundamentals):
        threshold = librosa.db_to_amplitude(threshold_db)
        compressed_magnitude = np.copy(magnitude)

        # Compute frequency bins (Hz) for each FFT bin
        freq_bins = np.linspace(0, self.sample_rate / 2, magnitude.shape[1])

        # Create a mask for fundamental frequencies and harmonics
        fundamental_mask = np.zeros_like(compressed_magnitude, dtype=bool)
        for f in fundamentals:
            for n in range(1, self.max_harmonics + 1):
                harmonic_bin = np.abs(freq_bins - (f * n)) < self.fundamental_tolerance
                fundamental_mask[:, harmonic_bin, :] = True

        # Apply compression
        for i in range(compressed_magnitude.shape[1]):  # Frequency bins
            for j in range(compressed_magnitude.shape[2]):  # Time frames
                for channel in range(compressed_magnitude.shape[0]):  # Stereo channels
                    original_value = compressed_magnitude[channel, i, j]

                    # Skip or process based on include_fundamentals flag
                    if include_fundamentals and not fundamental_mask[channel, i, j]:
                        continue
                    if not include_fundamentals and fundamental_mask[channel, i, j]:
                        continue

                    if original_value > threshold:
                        compressed_magnitude[channel, i, j] = (
                            threshold + (original_value - threshold) / ratio
                        )

        # Noise gate
        noise_threshold = librosa.db_to_amplitude(self.noise_threshold)
        compressed_magnitude[compressed_magnitude < noise_threshold] = 0

        # Smooth magnitude
        compressed_magnitude = self.smooth_magnitude(compressed_magnitude)

        return compressed_magnitude

    def calculate_auto_makeup_gain(self, original_magnitude, compressed_magnitude):
        noise_threshold = librosa.db_to_amplitude(-60)  # Ignore noise
        total_gain_reduction = 0
        non_zero_elements = 0

        for i in range(original_magnitude.shape[1]):
            for j in range(original_magnitude.shape[2]):
                for channel in range(original_magnitude.shape[0]):
                    if original_magnitude[channel, i, j] > noise_threshold:
                        gain_reduction = original_magnitude[channel, i, j] - compressed_magnitude[channel, i, j]
                        total_gain_reduction += gain_reduction
                        non_zero_elements += 1

        average_gain_reduction = (total_gain_reduction / non_zero_elements) if non_zero_elements > 0 else 0
        return max(min(1.0 + average_gain_reduction / 10, 1.1), 0.9)  # Clamp between 0.9 and 1.1

    def smooth_magnitude(self, magnitude):
        smoothed_magnitude = np.copy(magnitude)
        for i in range(1, magnitude.shape[1] - 1):
            smoothed_magnitude[:, i, :] = (
                0.25 * magnitude[:, i - 1, :] +
                0.5 * magnitude[:, i, :] +
                0.25 * magnitude[:, i + 1, :]
            )
        return smoothed_magnitude

    def reconstruct_audio(self, compressed_magnitude, phase):
        D_compressed = compressed_magnitude * phase
        processed_audio = np.array([librosa.istft(D_compressed[channel]) for channel in range(D_compressed.shape[0])])
        return processed_audio

    def save_audio(self, audio, output_file):
        sf.write(output_file, audio.T, self.sample_rate)

def detect_fundamental_with_limited_fft(audio, sr, low_freq=80, high_freq=400):
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    valid_range = (freqs >= low_freq) & (freqs <= high_freq)
    spectrum = spectrum[valid_range]
    freqs = freqs[valid_range]

    peak_index = np.argmax(spectrum)
    return [freqs[peak_index]] if len(freqs) > 0 else []


if __name__ == "__main__":
    # Define input/output folders
    current_directory = os.path.dirname(os.path.realpath(__file__))
    input_folder = os.path.join(current_directory, 'input')
    output_folder = os.path.join(current_directory, 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    # Process files
    plugin = DynamicEQPlugin()
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.wav', '.mp3')):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            audio, sr = plugin.load_audio(input_file)
            fundamentals = detect_fundamental_with_limited_fft(audio[0], sr)  # Detect fundamentals for the first channel
            processed_audio = plugin.process_audio(audio, fundamentals, include_fundamentals=True)
            plugin.save_audio(processed_audio, output_file)
            print(f"Processed: {input_file} -> {output_file}")
