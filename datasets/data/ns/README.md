# Navier-Stokes (PDEarena)
We download and preprocess files by hand surpassing hugging face, which requires git lfs. This gives a bit more leverage on which files to download and also is handy to have if your server doesn't have git lfs installed.

To download the data, run:
```sh
bash download.sh
```

From each file, we take pressure (scalar) and velocity (vector) fields and stack them. Each field is resized from 128 x 128 to 64 x 64 with anti-aliasing and converted to float32.
```sh
python preprocess.py
```