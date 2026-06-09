# Data and Generated Models

## Source

The academic results use the public CrisisMMD v2.0 release from QCRI:

- Dataset page: https://crisisnlp.qcri.org/crisismmd
- Paper: F. Alam, F. Ofli, and M. Imran, *CrisisMMD: Multimodal Twitter
  Datasets from Natural Disasters*, ICWSM 2018.

## Files intentionally excluded from Git

- `data/raw/`: downloaded annotations, 1.9 GB archive, and extracted images.
- `models/*.npy`: CLIP image/text embedding caches.
- `models/*.pkl`: fitted vectorizer and classifiers.
- `models/*.csv`: generated embedding metadata.

These files are large, reproducible, or distributed under the dataset's own
terms. They must not be committed to the coursework repository.

## Expected local layout

```text
data/raw/
  datasplit/crisismmd_datasplit_all/*.tsv
  CrisisMMD_v2.0/data_image/...
```

After placing the official files locally, rebuild with:

```powershell
python -m scripts.run_all
```

The repository keeps executed notebooks, processed audit manifests, metrics,
figures, screenshots, and the final PDF so the reported analysis remains
inspectable without uploading the raw image corpus.
