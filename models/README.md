# Models

`cw.onnx` is the one the application loads. The rest is provenance.

| file | what it is |
|---|---|
| `cw.onnx` | **shipped model.** 710 KB, 179,757 parameters, CNN + BiGRU + CTC. Exported from `cw.pt`. |
| `cw.pt` | the checkpoint `cw.onnx` was exported from. Byte-identical to `cw2.pt`. |
| `cw2.pt` | same file under the name it was trained as. 9000 steps, Farnsworth to 3.2:1. |
| `cw3.pt` | **not shipped.** The widened-Farnsworth experiment, kept because the negative result is worth keeping. |

## Why cw3 is here but not used

Training saw Farnsworth spacing only out to 3.2:1, and the ARRL 5 wpm file
measures a gap stretch of 16.6 -- a 6:1 send. The model had demonstrably never
seen the case it was failing on, so widening to 7:1 looked obviously right. It
made things worse, and worse on the target rather than merely elsewhere:

| | cw2 (shipped) | cw3 (widened) |
|---|---|---|
| ARRL 5 wpm | 53.3% | **50.0%** |
| ARRL 10 wpm | 62.7% | **55.7%** |
| synthetic | 97.7% | 96.4% |
| real, decoder chosen | 74.3% | 73.1% |

179,757 parameters over 9000 steps is a fixed budget, and spreading it across a
wider distribution buys coverage by spending depth. Anyone repeating this should
add capacity or steps before adding spread, and should oversample the extreme
rather than reach it with a thin tail.

## Reproducing

    python3 cwxss/train.py --steps 9000 --out models/cw.pt
    python3 cwxss/export.py models/cw.pt models/cw.onnx

About three hours on one CPU core. `torch` is needed to train and export;
inference needs only `onnxruntime`, which is why the shipped model is committed
-- a station in a field should not have to train one.
