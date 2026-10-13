# stitch-check: panorama from Sony RAWs with Vision + Metal

Experiment, not an eval. One Swift file (`main.swift`, ~660 lines, no dependencies, nothing taken from RapidRAW):
`CIRAWFilter` decode -> Vision registration of every pair -> homographies chained to a reference frame ->
one Metal compute pass that warps and feather-blends. RAWs are read in place on the T7 drive; nothing is written there.

**Short version:** it works and it is fast (about 2 s for four 24 MP RAWs at full resolution, 72 MP output),
the seams are clean on a distant landscape, but **Vision's registration alone cannot align a pan**. It needed a
coarse shift estimate first (my own code, not Vision). Details below.

## Frames used

Main set, `/Volumes/T7/CROSS_COUNTRY_2026/RAWS/`, ILCE-7M3 + FE 85mm F1.8, handheld, 8 s start to finish,
identical exposure (1/320, f/10, ISO 100, auto WB):

| File | Time (EXIF) | Position in pan |
|---|---|---|
| DSC09304.ARW | 13:56:52 | far left |
| DSC09303.ARW | 13:56:50 | left |
| DSC09305.ARW | 13:56:56 | right (chosen as reference) |
| DSC09306.ARW | 13:56:58 | far right |

Second set to see a harder case, same folder, APS-C crop 35 mm, handheld, 9 s: DSC09632 to DSC09635.
Nearer foreground (parked cars, trees), a moving car, and the last frame shot at a different exposure
(1/60 f/4 ISO 160 against 1/500 f/2.8 ISO 400).

How they were found: EXIF scan of the T7 shoots for runs of 3 to 6 frames, each within 5 s of the previous,
same focal length and orientation, then contact sheets of the embedded previews, then the tool's own overlap check.
Most such runs turned out to be re-shoots of one composition, not pans.

## Outputs (`out/`)

| File | What |
|---|---|
| `prairie-preview.jpg` | 2000 px preview of the full-resolution stitch |
| `prairie-seam1.jpg`, `-seam2`, `-seam3` | 1600x1000 crops at 100 %, centred where two frames carry equal weight |
| `prairie-half-full.jpg` | complete stitch from half-size decode, 7028x2581 |
| `prairie-full.jpg` | complete full-resolution stitch, 14051x5162, 12 MB. Not committed; regenerate below |
| `lot-preview.jpg`, `lot-seam1..3.jpg` | second set |
| `*-metrics.json` | raw numbers for each run |

## Numbers

Mac16,7, Apple M4 Pro, 24 GB, macOS 27.0.1. Wall-clock seconds, range over 2 to 3 warm runs.
"Cold" is the first run after build: RAWs not yet in the file cache (external SSD) and Core Image not warmed up.

| | Prairie, full res | Prairie, half | Lot, full res |
|---|---|---|---|
| Frames | 4 x 6000x4000 | 4 x 3000x2000 | 4 x 3936x2624 |
| Output canvas | 14051x5162 (72.5 MP) | 7028x2581 (18.1 MP) | 6733x3122 (21.0 MP) |
| Decode, all frames | 0.95 to 1.01 (cold 4.27) | 0.83 to 0.87 (cold 4.23) | 0.65 (cold 1.96) |
| Registration, 6 pairs | 1.03 to 1.11 | 1.07 to 1.13 | 1.09 to 1.16 |
| Warp + blend | 0.042 to 0.073 | 0.006 to 0.017 | 0.010 to 0.030 |
| Whole pipeline | 2.13 to 2.28 (cold 5.56) | 2.01 to 2.09 (cold 5.48) | 1.86 to 2.22 (cold 3.19) |
| Full JPEG encode (extra) | 0.06 to 0.07 | 0.02 | 0.02 |
| Peak memory (`phys_footprint` peak) | 2.14 to 2.45 GB | 1.07 to 1.21 GB | 1.61 to 1.62 GB |
| Peak memory (`/usr/bin/time -l`) | 2.24 to 2.56 GB | 1.12 to 1.27 GB | 1.69 to 1.70 GB |

Notes on the numbers:

- Registration always runs on 2048 px copies, so it costs the same at any working size. Of the ~1.1 s:
  Vision calls 0.33 to 0.37 s (two requests on the whole frames and two on the overlap crops, per pair),
  coarse shift search 0.48 s, scoring candidates 0.22 to 0.26 s, plus 0.10 to 0.13 s making the 2048 px copies.
  It registers all 6 pairs so shooting order does not matter; adjacent pairs only would be about half.
- Decoding at half size is barely faster than full size (0.85 s against 0.97 s), so there is little reason to
  work at reduced size on this machine.
- The warp + blend pass is not the bottleneck at any size: 72.5 MP in about 50 ms (8 to 13 ms of GPU time).
  The shader compiled in 0.36 s the first time and is cached after that.
- Memory at full resolution is mostly four 6000x4000 half-float frames (192 MB each) plus the 14051x5162
  half-float canvas (580 MB), and Core Image's own working memory on top.

## What registration actually took

- Vision on whole frames (`VNHomographicImageRegistrationRequest`, then `VNTranslationalImageRegistrationRequest`)
  failed on **every** pair of the prairie pan: 0 of 6 usable. It returned confident-looking but wrong
  near-identity transforms, with no error and no confidence value to tell you so. It only worked directly on
  near-duplicate frames (about 95 % overlap or more), which is what it is built for.
- What made it work: a brute-force search for the best integer shift on 128 px high-passed thumbnails
  (my code), crop both frames to the overlap that shift implies, then run Vision's homographic request on the
  crops. With that head start Vision refined every overlapping pair to a good homography (4 of 6 prairie pairs;
  the other 2 really are only 12 % overlap and were correctly rejected).
- Because Vision gives no quality signal, the tool scores every result itself (correlation of high-passed
  luminance inside the claimed overlap) and rejects below 0.3. Accepted prairie pairs scored 0.41 to 0.51,
  lot pairs 0.36 to 0.79. Plain luminance correlation was useless for this: sky over grass correlates at
  0.5 to 0.8 under any shift.
- Sets that did not stitch at all: a river series shot from a moving car (no usable pair) and a skyline
  series where the photographer moved on between frames 3 and 4 (no overlap). Both were rejected, not faked.

## What it looks like

Prairie (the committed crops):

- No visible seam lines, steps in brightness or colour, or doubled edges in any of the three seam crops.
  Compared at 200 % against each single frame rendered alone (`--solo`), the blend is about as sharp as either
  source: fence posts, the corral and the tree line land within roughly a pixel.
- Exposure: EXIF exposure was identical, yet the overlap means still differed, so the per-frame gains came out
  at 0.91 to 1.09. Likely vignetting (overlaps compare one frame's edge with another's centre). With gains on,
  no step is visible; I did not render a gains-off comparison.
- Sensor dust: the same dust spots repeat across the sky once per frame (clearly visible in the preview).
  That is the camera, but a stitch makes it obvious.
- The outline is the raw union of the warped frames with black corners; there is no auto-crop.
- Grass in the near foreground moved in the wind between frames; it reads as slightly soft in the overlaps,
  not as ghosting.

Lot (harder):

- Aligns and looks right at preview size, including the frame shot 1.3 stops differently (gain 0.73 applied).
- At 100 % the overlaps are visibly softer than a single frame, and the row of parked cars at the bottom of
  `lot-seam1.jpg` shows a faint ghost car. This is parallax from nearby objects, which one homography per
  frame cannot fix, made visible by the feather blend averaging both frames. The moving car on the road is soft
  for the same reason. A seam-finding or multiband blend would hide most of this; neither is implemented.

## Limits of this experiment

- Feather blend only. No multiband, no seam finding, no ghost removal.
- Planar projection only. Fine for these pans (roughly 50 to 60 degrees across), wrong for wide ones, which need a
  cylindrical or spherical projection.
- No bundle adjustment: homographies are chained along a spanning tree, so error would accumulate over many frames.
- White balance is copied from the first frame to the rest; lens corrections are whatever `CIRAWFilter` does by default.
- Two sets, one machine, no ground truth. Quality statements are from looking at crops.
- The coarse shift search assumes roughly a pure shift between frames (no large rotation or zoom).

## Reproduce

```bash
swiftc -O -swift-version 5 main.swift -o stitch-check
```

```bash
R=/Volumes/T7/CROSS_COUNTRY_2026/RAWS; ./stitch-check --out out --name prairie --scale 1.0 $R/DSC09303.ARW $R/DSC09304.ARW $R/DSC09305.ARW $R/DSC09306.ARW
```

Options: `--scale` (decode size, default 0.5), `--reg-edge` (default 2048), `--power` (feather exponent, default 2),
`--no-gain`, `--no-lock-wb`, `--solo N` (draw one frame only), `--crops N`.
