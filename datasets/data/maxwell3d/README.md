# Maxwell (PDEarena)
We download and preprocess files by hand surpassing hugging face, which requires git lfs. This gives a bit more leverage on which files to download and also is handy to have if your server doesn't have git lfs installed.

To download the data, run:
```sh
bash download.sh
```

From each file, we take displacement D (vector) and magnetic (bivector) fields and concatenate them. Each field is converted to float32.
```sh
python preprocess.py
```