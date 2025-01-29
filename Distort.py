# Librosa Distortion + Xavier's Dyanmic Compressor

'''
  TODO
  Load audio
  Compress w/ compress.py -- process_audio() -- Compress before or after?
  Take again and amplify
  Clip the magnitude
  Output audio
'''

import compress as cmp


def distort_audio(eqPlugin, audio):
    magnitude, phase = eqPlugin.get_spectrum(audio)

    # TODO 
    # Amplify magnitude
    # Clip magnitude at threshold

# Main loop
if __name__ == "__main__":
    # Define input/output folders
    current_directory = os.path.dirname(os.path.realpath(__file__))
    input_folder = os.path.join(current_directory, 'input')
    output_folder = os.path.join(current_directory, 'output')
    os.makedirs(output_folder, exist_ok=True)

    eq = cmp.DyanmicEQPlugin()

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.wav', '.mp3')):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            audio, sr = eq.load_audio(input_file)

            fundamentals = detect_fundamental_with_limited_fft(audio[0], sr)

            #distorted_audio = distort_audio(eq, audio)
            #compressed_audio = eq.process_audio(distorted_audio, fundamentals, include_fundamentals=True)
