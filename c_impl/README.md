# GatedDeltaNet C Reference

This directory contains a deliberately simple pure-C reference path for the
`m-a-p/1.3B-100B-GatedDeltaNet-pure` checkpoint. It is structured as a flat
weight dump, a forward-only model file, and a separate evaluation testbench.

## Files

- `gdn_model.c`: forward-only model implementation
- `gdn_model.h`: model types and function declarations
- `gdn_eval.c`: separate evaluation testbench
- `Makefile`: builds the testbench without BLAS
- `artifacts/*.gdnw`: flat float32 weight export
- `fixtures/*.gdnreq`: pretokenized evaluation requests

## Build

```bash
make -C c_impl
```

## Export Weights

```bash
/ibex/user/yaoz0b/conda-environments/GatedDeltaNet/bin/python \
  scripts/export_gdn_c.py weights \
  --output c_impl/artifacts/gdn-1.3b-f32.gdnw
```

## Export Eval Fixtures

```bash
/ibex/user/yaoz0b/conda-environments/GatedDeltaNet/bin/python \
  scripts/export_gdn_c.py fixtures \
  --tasks piqa hellaswag winogrande arc_easy arc_challenge social_iqa boolq lambada_openai wikitext \
  --output-dir c_impl/fixtures
```

Use `--limit` to create a smaller parity or smoke set.

## Run C Eval

```bash
./c_impl/gdn_eval \
  c_impl/artifacts/gdn-1.3b-f32.gdnw \
  c_impl/fixtures/piqa.gdnreq \
  c_impl/results/piqa_c.json
```

## Run Python Reference

```bash
srun --jobid 46223467 bash -lc '
  cd /home/yaoz0b/GatedDeltaNet &&
  /ibex/user/yaoz0b/conda-environments/GatedDeltaNet/bin/python \
    scripts/compare_gdn_c.py \
    --fixture c_impl/fixtures/piqa.gdnreq \
    --output c_impl/results/piqa_python.json \
    --device cuda \
    --dtype float32
'
```

## Compare C vs Python

```bash
/ibex/user/yaoz0b/conda-environments/GatedDeltaNet/bin/python \
  scripts/check_gdn_c_parity.py \
  --python-dir c_impl/results_smoke_python \
  --c-dir c_impl/results_smoke_c \
  --output c_impl/results_smoke_parity.json
```

The intended workflow is:

1. Export the checkpoint into a flat weight file once.
2. Export tokenized evaluation fixtures.
3. Run the same fixture set through the Python reference and the C runtime.
4. Compare the JSON outputs to confirm parity before moving to HLS-oriented work.
