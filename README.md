# To use:
create a new conda env (`conda create --name $name$`) then activate (using `conda activate $name$`)
and run `pip install -r requirements.txt`

to use a specific type of python you can also use a python env via venv
`python<version> -m venv venv`
then `venv/Scripts/activate` or `venv/bin/activate` (if on Mac)

place any number of songs (.wav, .mp3's both work)
into the 'input' folder

run dyn compressor via `python compress.py` or `python compress2.py`

It will run it and smooth out peaking frequencies via ratio using FFT

I created 2 different ones that are very similar just to test out some different settings.
Settings can be adjusted in the `compress.py` or `compress2.py` files inside the `bands` object.

`test.py` generates some sine waves and attempts to display the before and after, but I'm still working on it.
___________________________________________

Notes:
Can we potentially use pink noise or white noise as a target instead of having to run FFT constantly?