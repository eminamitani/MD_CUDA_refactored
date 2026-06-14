# Model files

Place trained neural-network potential artifacts here. The AOTI example config
expects:

```text
models/model_schnet_aoti.pt2
```

TorchScript models can be used by setting the potential type to `NNP` or
`NNP_csr` and pointing `model_path` to the corresponding artifact.
