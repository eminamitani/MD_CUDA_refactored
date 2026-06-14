# Input structures

Place extended XYZ input structures here. `configs/example_workflow_NVE_NNP_aoti.json`
expects `data/sample_NS2.xyz` by default.

The reader expects the second line to include a `Lattice="..."` field and each
atom line to contain:

```text
species x y z fx fy fz
```
