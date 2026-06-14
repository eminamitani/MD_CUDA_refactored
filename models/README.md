# Model files

Place trained neural-network potential artifacts here. The AOTI example config
expects:

```text
models/model_schnet_aoti.pt2
```

TorchScript models can be used by setting the potential type to `NNP` or
`NNP_csr` and pointing `model_path` to the corresponding artifact.

For simplegnn PaiNN checkpoints, export a TorchScript wrapper first:

```sh
python scripts/export_simplegnn_painn_for_md.py \
  --simplegnn-root /path/to/simplegnn_version2 \
  --checkpoint /path/to/painn_model.pth \
  --output models/deployed_painn_model.pt
```
