# Mask check: Apple Vision vs SAM 2.1 (Core ML)

A quick visual check, not a formal eval. 25 Sony ARWs (ILCE-7M3; 8 portraits, 5 groups, 8 objects/details,
4 landscapes) decoded with `CIRAWFilter` to a 2048 px long edge (all 25 decoded as RAW, no preview fallback),
then masked four ways:

- **Vision person**: `VNGeneratePersonSegmentationRequest`, `.accurate`
- **Vision subject**: `VNGenerateForegroundInstanceMaskRequest`, all instances
- **Vision attention saliency**: heatmap only; its peak is the SAM prompt
- **SAM 2.1 small**: `apple/coreml-sam2.1-small` (Apache-2.0, float16), **one positive point** at the largest
  face centre if Vision finds a face (12 images), otherwise the saliency peak (13 images). Two readouts of the same
  run: the *best-score* mask (SAM's default pick) and the *largest* of its three candidates.

Open `index.html` for the grid; `sheets/` has one 3x2 sheet per image (800 px panels); `results.csv` has per-image numbers
for all 25.

**Only 9 of the 25 sheets are in this repo** (14, 15, 16, 17, 18, 20, 22, 23, 25: hands, food, statue, swan,
landscapes). The other 16 show recognisable people or a readable number plate and stay on the local machine, so
sheet numbers cited below that are not in that list cannot be opened here.

Machine: MacBook Pro Mac16,7, Apple M4 Pro, 24 GB, macOS 27.0.1.

## Read on quality

I looked closely at 14 of the 25 sheets (across all four kinds) and used mask coverage in `results.csv` for the rest.

- **People: Vision wins, clearly.** Person and subject masks cover the whole figure with clean edges, including
  hair and fingers (01, 03, 06, 08). SAM from one face point returns just the face on all 12 face-prompted images
  (best-score coverage 0.4-12 %). Its largest candidate is usually the whole person (01, 02, 06, 08) but with a low
  predicted IoU, so you cannot select it by score, and its edges are visibly blockier (256 px mask upsampled).
- **Vision person vs subject:** person segmentation takes every human in frame, including out-of-focus people behind
  and beside the subject (03, 06); subject lift keeps only the foreground figure there, which is closer to
  "the subject". In the big group shot (10) subject covers the whole group more completely than person.
- **Vision person false positives:** the bronze statue (17, 56 %) and the foot-on-plate detail (18, 80 %, most of
  the floor). It is correct on hands (14) and on silhouetted bystanders in the skyline (24).
- **Objects: Vision subject is the most useful zero-prompt mask.** Car (19), swan (20), elk + car (21), hands (14)
  are all tight. It does choose its own subject: on the buffet table (16) it lifts the bread board, not the cakes
  the saliency peak is on.
- **SAM is at its best on a single compact object with a good point:** elk (21), lone tree (23), tower (24) are
  tight and correct, and the tree and tower are things Vision returns nothing for.
- **SAM failures from one point:** sub-part masks (one finger on 14, the tail-light bar on 19, one cake slice
  on 16), and the swan (20), where the saliency peak landed between the two necks and SAM confidently (iou 0.99)
  masked the water instead.
- **Landscapes:** Vision subject returns 0 instances on three of four (the fourth, Mount Rushmore, gets 2 and a
  "face" prompt), which is the honest answer. SAM always returns something; its largest candidate there is sky or
  hillside.

Bottom line: for an automatic "where is the subject" mask, Vision subject (with person as a people-only signal) is
better and far cheaper. SAM 2.1 only earns its cost as an interactive click-to-mask tool, or with a box prompt
derived from the Vision mask; one automatic point is not enough to pick the right granularity.

## Speed (warm, median of 3 runs per image, median over 25 images)

| step | ms |
|---|---|
| RAW decode to 2048 px (`CIRAWFilter`) | 621 |
| Vision person (accurate) | 34 |
| Vision subject (incl. scaled mask) | 19 |
| Vision attention saliency | 5.9 |
| Vision face rectangles | 5.5 |
| SAM 2.1 image encoder, compute units `.all` | 178 |
| SAM 2.1 prompt encoder + mask decoder | 6.4 |
| **SAM 2.1 total** | **185** |

SAM encoder by compute units (single warm call per image, median): `.cpuAndGPU` 80 ms, `.all` 141 ms,
`.cpuOnly` 154 ms (+39 ms decoder). GPU was the fastest option on this machine.

Cold / first run:

- Vision person first call: 106-108 ms in these runs; 1.65 s on the very first run of the session.
- Vision subject first call: 109-137 ms. Saliency 24-32 ms, face 34-38 ms.
- SAM model load (compiled `.mlmodelc`, `.all`): about 1.7-1.8 s, nearly all in the mask decoder; 3.65 s the first
  time ever on this Mac. With `.cpuAndGPU` load is 0.18 s.
- SAM first prediction after load: between 0.25 s + 0.08 s and 3.2 s + 2.3 s (encoder + decoder) across runs.
- One-time `coremlcompiler` compile of the three packages is not included.

## Memory (separate process per method, all 25 images)

| run | peak RSS (MB) | peak phys footprint (MB) |
|---|---|---|
| decode only | 219 | 1654 |
| + Vision person | 308 | 1510 |
| + Vision subject | 272 | 1675 |
| + saliency + face | 265 | 1659 |
| + SAM 2.1 (`.all`) | 422 | 2360 |
| + SAM 2.1 (`.cpuAndGPU`) | 473 | 2034 |
| + SAM 2.1 (`.cpuOnly`) | 680 | 1861 |
| everything in one process | 548 | 1953 |

The footprint column is dominated by `CIRAWFilter` decode and Core Image caches (1.5-1.7 GB with no masking at
all), so it is noisy. Roughly: each Vision request adds 50-90 MB of RSS; SAM 2.1 small adds about 200-460 MB of RSS
and about 0.2-0.7 GB of footprint. Model files on disk: 93 MB for SAM 2.1 small; Vision ships with the OS.

## Caveats

- One point per image and no box prompt, so this is SAM's hardest setting; a box from the Vision mask would do better.
- SAM masks are the 256 px low-res output, sigmoid then bilinear upsample; no refinement.
- Timings are wall time on an otherwise idle machine, not a controlled benchmark.

## Reproduce

```
cd tool
swiftc -O -swift-version 5 main.swift -o maskcheck
# images.tsv here lists kind + file name only; the tool wants kind<TAB>full path to each ARW
# models: https://huggingface.co/apple/coreml-sam2.1-small, each .mlpackage compiled with `xcrun coremlcompiler compile`
./maskcheck --list images.tsv --out .. --models /path/to/compiled/models
./maskcheck --list images.tsv --out .. --models /path/to/compiled/models --only sam --units cpuAndGPU   # memory/units runs
python make_index.py   # from mask-check/
```

No RAWs or model weights are in this folder. No RapidRAW code was used.
