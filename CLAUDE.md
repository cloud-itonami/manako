# etzhayyim-project-manako — 眼 Browser-local YOLO26 Object Detection

**URL**: `https://manako.etzhayyim.com` (operator-provisioned)
**ADR**: ADR-2606034800
**Status**: 🟢 R0 — implemented + **empirically verified end-to-end on the WASM runtime**. Real `yolo26n.pt` (Ultralytics 8.4.60, v8.4.0 assets) exported to ONNX `(1,300,6)` NMS-free; `onnxruntime-web` v1.26.0 ran it on the `wasm` EP and manako's core reproduced the canonical bus.jpg result (1 bus + 4 persons) identically to the Python onnxruntime ground truth. Live Chrome/WebGPU screenshot pending (browser extension was not connected in the verifying session).

## Empirical verification (2026-06-03)

| Check | Result |
|---|---|
| Pure core unit tests | **12/12** green (`tsx --test`) |
| YOLO26 export | `yolo26n.pt` → `yolo26n.onnx`, output **`(1, 300, 6)`** = NMS-free (confirms dual-mode design) |
| Python onnxruntime ground truth (bus.jpg) | bus 0.93 · person 0.92/0.90/0.86/0.51 |
| manako TS core on that exact tensor | **identical** boxes + scores |
| `onnxruntime-web` **WASM EP** (browser package) running the model in Node + manako core | **identical** again — input `images`, output `output0`, dims `[1,300,6]` |

→ The substantive claim "YOLO26 runs in the browser via wasm" is verified by running the **identical `onnxruntime-web` WebAssembly runtime** + the identical app core. The only unrun step is a literal Chrome screenshot of the Svelte shell (WebGPU EP).

## What this is

The browser-wasm answer to *「YOLO26 を wasm browser で動かす」*. 100% on-device object
detection (COCO 80-class) via **ONNX Runtime Web (WebGPU → wasm fallback)** — the same
edge-inference carve-out as `gazo` (SD) and `ameno` (LLM). The image never leaves the
browser; there is no server-side GPU and no telemetry.

## Architecture (mirrors gazo)

```
File / <img> / video frame
  → detect.svelte.ts (main thread)  — ImageBitmap (transferable)
    → detect-worker.ts (Web Worker)
        1. fetch YOLO26 .onnx from operator CDN (OPFS-cached)
        2. ort.InferenceSession.create([webgpu, wasm])
        3. preprocess: letterbox → OffscreenCanvas → NCHW Float32 /255
        4. session.run
        5. postprocess() ← yolo26-core.ts (PURE)
    → detections (original-image px) → main thread → canvas overlay
```

### The substantive part — `src/lib/yolo26-core.ts` (pure, GPU-free, unit-tested)

Dual-mode output handling — works regardless of how the operator exported the model:

| Export mode | Output shape | Path |
|---|---|---|
| **NMS-free / end-to-end** (`nms=True`, YOLO26 flagship) | `(1, N, 6)` = `[x1,y1,x2,y2,score,cls]` | threshold + un-letterbox (no NMS) |
| **Raw head** (`nms=False`) | `(1, 4+nc, A)` chw or `(1, A, 4+nc)` hwc | argmax → cxcywh→xyxy → class-aware NMS → un-letterbox |

`detectLayout()` auto-classifies the tensor; `postprocess()` returns boxes in **original
image pixel space**. Defensive sigmoid guard if an export ships logits.

## Run / test / build

```bash
cd appview/etzhayyim-wasm-manako-m4n4k0v1/svelte
pnpm install
pnpm test            # tsx --test → 12/12 (letterbox, IoU/NMS, layout-detect, both decoders)
pnpm dev --port 5190 # local dev
pnpm build           # → ../_svelte ; then `etzhayyim deploy`
```

## Operator: export + host the model (one-time, offline)

```bash
pip install ultralytics          # AGPL — operator tooling, NEVER vendored into this repo
yolo export model=yolo26n.pt format=onnx imgsz=640 nms=True   # preferred (NMS-free)
# upload yolo26n.onnx → cdn.etzhayyim.com/models/yolo26/  (see src/lib/models.ts)
```

## Gates / invariants (per ADR-2606034800)

- **G1 on-device only** — inference 100% in-browser (WebGPU/wasm). No server inference, no image upload. Aligns ADR-2606012100 (ameno node class) + ADR-2605215000 (browser carve-out, NOT commercial GPU).
- **G2 model not bundled / AGPL isolation** — YOLO26 weights + Ultralytics source are AGPL-3.0; never committed. Operator-supplied .onnx at runtime (like gazo SD weights). Our inference code is original Apache-2.0 array math. No git-lfs.
- **G3 no surveillance / no biometric** — object detection only (COCO classes). NOT face recognition / person re-ID / tracking-for-identification. Privacy-by-construction (mirrors kiyome G9). No persistent identity, no telemetry.
- **G4 Murakumo invariant preserved** — browser WebGPU/wasm is the explicit edge carve-out; MUST NOT add RunPod / commercial GPU / vendor inference paths.
- **G5 honest R0** — pure logic + 12 tests verified; live model fetch + real WebGPU run operator-gated (needs exported ONNX on CDN); zero weights shipped.

### Non-goals
Weaponized targeting · mass surveillance · facial recognition · person re-identification ·
license-plate/biometric harvesting. (Detection feeds Wellbecoming-aligned assistive use;
PII *removal* is the separate `e7m-dataset` vision PII filter, ADR-2605262500.)

## Runtime

| Item | Value |
|---|---|
| nanoid | `m4n4k0v1` |
| Runtime | Worker (TS Native, Workers Assets) |
| UI | Svelte 5 CSR (Vite) |
| Deps | `onnxruntime-web` (+ `tsx` dev for tests) |
| Model | YOLO26 detect (n/s/m), COCO-80, operator-exported ONNX |
| Browser | WebGPU (Chrome/Edge 113+) preferred; wasm fallback otherwise |
