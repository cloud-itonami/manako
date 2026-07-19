/**
 * yolo26-core.ts — Pure, framework-free YOLO26 pre/post-processing core.
 *
 * This module is the **substantive, verifiable** part of manako (眼) browser-local
 * object detection. It contains ZERO DOM / ZERO onnxruntime imports so it runs and
 * is unit-tested under plain Node (`node --import tsx --test`). The GPU/wasm session
 * and DOM image plumbing live in `detect-worker.ts`.
 *
 * **Why dual-mode?** A YOLO26 ONNX model can be exported in two shapes and we must
 * handle either without knowing in advance:
 *
 *   1. NMS-free / end-to-end (`nms=True`, YOLO26's flagship mode):
 *      output `(1, N, 6)` rows already decoded + de-duplicated → `[x1,y1,x2,y2,score,cls]`
 *      (model-input pixel space). No NMS needed — just threshold + un-letterbox.
 *
 *   2. Raw (`nms=False`, classic head): output `(1, 4+nc, A)` ("chw") or `(1, A, 4+nc)`
 *      ("hwc") — 4 box channels (cxcywh, input px) + nc per-class scores. Needs
 *      argmax → threshold → cxcywh→xyxy → NMS → un-letterbox.
 *
 * Output boxes are always returned in ORIGINAL image pixel coordinates.
 *
 * License: Apache-2.0 + etzhayyim Charter Compliance Rider v2.0. This is original
 * array math, NOT derived from Ultralytics source (which is AGPL-3.0). The YOLO26
 * model weights are a separate runtime artifact the operator supplies (see models.ts).
 *
 * @module
 */

/** Axis-aligned box, pixel coordinates: `[x1, y1, x2, y2]`. */
export type BBox = [number, number, number, number];

/** A single detection in ORIGINAL image pixel space. */
export interface Detection {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
	score: number;
	classId: number;
	label: string;
}

/** Letterbox transform mapping ORIGINAL image → square model input and back. */
export interface LetterboxParams {
	/** Uniform resize factor applied to the source before padding. */
	scale: number;
	/** Left padding in model-input pixels. */
	padX: number;
	/** Top padding in model-input pixels. */
	padY: number;
	inputW: number;
	inputH: number;
	srcW: number;
	srcH: number;
}

export interface PostprocessOptions {
	/** Number of object classes (COCO = 80). */
	numClasses: number;
	/** Class label table (index → name). */
	labels: readonly string[];
	/** Confidence threshold. Default 0.25 (Ultralytics default). */
	confThreshold?: number;
	/** IoU threshold for NMS (raw layout only). Default 0.45. */
	iouThreshold?: number;
	/** Maximum detections returned. Default 300. */
	maxDetections?: number;
}

export type OutputLayout = 'nms-free' | 'chw' | 'hwc';

/* ── Letterbox geometry ───────────────────────────────────────────────── */

/**
 * Compute the aspect-preserving letterbox transform (Ultralytics convention:
 * resize by min-scale, center-pad with neutral gray). Pure geometry — the actual
 * pixel resample happens on an OffscreenCanvas in the worker using these params.
 */
export function computeLetterbox(
	srcW: number,
	srcH: number,
	inputW: number,
	inputH: number,
): LetterboxParams {
	const scale = Math.min(inputW / srcW, inputH / srcH);
	const newW = Math.round(srcW * scale);
	const newH = Math.round(srcH * scale);
	const padX = Math.floor((inputW - newW) / 2);
	const padY = Math.floor((inputH - newH) / 2);
	return { scale, padX, padY, inputW, inputH, srcW, srcH };
}

/** Map a box from model-input pixel space back to ORIGINAL image space (clamped). */
export function unletterboxBox(b: BBox, p: LetterboxParams): BBox {
	const x1 = clamp((b[0] - p.padX) / p.scale, 0, p.srcW);
	const y1 = clamp((b[1] - p.padY) / p.scale, 0, p.srcH);
	const x2 = clamp((b[2] - p.padX) / p.scale, 0, p.srcW);
	const y2 = clamp((b[3] - p.padY) / p.scale, 0, p.srcH);
	return [x1, y1, x2, y2];
}

/* ── IoU + NMS ────────────────────────────────────────────────────────── */

/** Intersection-over-union of two xyxy boxes. */
export function iou(a: BBox, b: BBox): number {
	const ix1 = Math.max(a[0], b[0]);
	const iy1 = Math.max(a[1], b[1]);
	const ix2 = Math.min(a[2], b[2]);
	const iy2 = Math.min(a[3], b[3]);
	const iw = Math.max(0, ix2 - ix1);
	const ih = Math.max(0, iy2 - iy1);
	const inter = iw * ih;
	const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
	const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
	const union = areaA + areaB - inter;
	return union <= 0 ? 0 : inter / union;
}

/**
 * Greedy non-maximum suppression. Returns kept indices, sorted by descending score.
 * Class-aware: only suppresses boxes of the SAME class (Ultralytics `agnostic=False`).
 */
export function nms(
	boxes: BBox[],
	scores: number[],
	classes: number[],
	iouThreshold: number,
	maxDetections: number,
): number[] {
	const order = scores
		.map((s, i) => i)
		.sort((a, b) => scores[b] - scores[a]);
	const kept: number[] = [];
	const removed = new Uint8Array(boxes.length);
	for (const i of order) {
		if (removed[i]) continue;
		kept.push(i);
		if (kept.length >= maxDetections) break;
		for (const j of order) {
			if (removed[j] || j === i) continue;
			if (classes[j] !== classes[i]) continue;
			if (iou(boxes[i], boxes[j]) > iouThreshold) removed[j] = 1;
		}
	}
	return kept;
}

/* ── Layout detection ─────────────────────────────────────────────────── */

/**
 * Classify the raw ONNX output tensor shape into a known YOLO26 layout.
 *
 * - `(1, N, 6)`           → 'nms-free'  (feature dim 6 ≠ 4+nc)
 * - `(1, 4+nc, A)`        → 'chw'       (channel axis = dim 1)
 * - `(1, A, 4+nc)`        → 'hwc'       (channel axis = dim 2)
 */
export function detectLayout(dims: readonly number[], numClasses: number): OutputLayout {
	if (dims.length !== 3) {
		throw new Error(`unexpected output rank ${dims.length} (dims=${dims.join('x')})`);
	}
	const feat = 4 + numClasses;
	const d1 = dims[1];
	const d2 = dims[2];
	// NMS-free rows are 6 wide (x1,y1,x2,y2,score,cls) and won't equal 4+nc for nc>2.
	if (d2 === 6 && d1 !== feat) return 'nms-free';
	if (d1 === 6 && d2 !== feat) return 'nms-free';
	if (d1 === feat) return 'chw';
	if (d2 === feat) return 'hwc';
	throw new Error(
		`cannot infer YOLO26 layout from dims=${dims.join('x')} with numClasses=${numClasses}`,
	);
}

/* ── Decoders ─────────────────────────────────────────────────────────── */

/** Logistic guard: Ultralytics bakes sigmoid into the graph, but if an export ships
 * raw logits (values well outside [0,1]) we activate defensively. */
function maybeSigmoid(v: number): number {
	if (v >= -0.001 && v <= 1.001) return v;
	return 1 / (1 + Math.exp(-v));
}

function clamp(v: number, lo: number, hi: number): number {
	return v < lo ? lo : v > hi ? hi : v;
}

interface RawCandidates {
	boxes: BBox[];
	scores: number[];
	classes: number[];
}

/**
 * Decode a raw `(1, 4+nc, A)` / `(1, A, 4+nc)` tensor into thresholded candidates
 * (model-input pixel space, xyxy). NMS is applied separately by `postprocess`.
 */
export function decodeRaw(
	data: Float32Array | number[],
	dims: readonly number[],
	layout: 'chw' | 'hwc',
	numClasses: number,
	confThreshold: number,
): RawCandidates {
	const feat = 4 + numClasses;
	const numAnchors = layout === 'chw' ? dims[2] : dims[1];
	// Accessor: value at anchor `a`, feature channel `c`.
	const at =
		layout === 'chw'
			? (a: number, c: number) => data[c * numAnchors + a]
			: (a: number, c: number) => data[a * feat + c];

	const boxes: BBox[] = [];
	const scores: number[] = [];
	const classes: number[] = [];

	for (let a = 0; a < numAnchors; a++) {
		// argmax over class scores
		let best = -Infinity;
		let bestId = -1;
		for (let c = 0; c < numClasses; c++) {
			const s = maybeSigmoid(at(a, 4 + c));
			if (s > best) {
				best = s;
				bestId = c;
			}
		}
		if (best < confThreshold) continue;
		const cx = at(a, 0);
		const cy = at(a, 1);
		const w = at(a, 2);
		const h = at(a, 3);
		boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
		scores.push(best);
		classes.push(bestId);
	}
	return { boxes, scores, classes };
}

/* ── Full postprocess ─────────────────────────────────────────────────── */

/**
 * Convert a raw ONNX output tensor into final detections in ORIGINAL image space.
 * Auto-detects the layout and applies the correct path (passthrough for NMS-free,
 * decode+NMS for raw).
 */
export function postprocess(
	data: Float32Array | number[],
	dims: readonly number[],
	lb: LetterboxParams,
	opts: PostprocessOptions,
): Detection[] {
	const conf = opts.confThreshold ?? 0.25;
	const iouT = opts.iouThreshold ?? 0.45;
	const maxDet = opts.maxDetections ?? 300;
	const layout = detectLayout(dims, opts.numClasses);

	let kept: { box: BBox; score: number; classId: number }[];

	if (layout === 'nms-free') {
		// Rows already de-duplicated; each row = [x1,y1,x2,y2,score,cls].
		const rowMajorFirst = dims[1] !== 6; // (1,6,N) vs (1,N,6)
		const n = rowMajorFirst ? dims[1] : dims[2];
		const get = rowMajorFirst
			? (i: number, k: number) => data[i * 6 + k]
			: (i: number, k: number) => data[k * n + i];
		kept = [];
		for (let i = 0; i < n; i++) {
			const score = get(i, 4);
			if (score < conf) continue;
			kept.push({
				box: [get(i, 0), get(i, 1), get(i, 2), get(i, 3)],
				score,
				classId: Math.round(get(i, 5)),
			});
		}
		kept.sort((a, b) => b.score - a.score);
		kept = kept.slice(0, maxDet);
	} else {
		const cand = decodeRaw(data, dims, layout, opts.numClasses, conf);
		const keepIdx = nms(cand.boxes, cand.scores, cand.classes, iouT, maxDet);
		kept = keepIdx.map((i) => ({
			box: cand.boxes[i],
			score: cand.scores[i],
			classId: cand.classes[i],
		}));
	}

	return kept.map((d) => {
		const [x1, y1, x2, y2] = unletterboxBox(d.box, lb);
		return {
			x1,
			y1,
			x2,
			y2,
			score: d.score,
			classId: d.classId,
			label: opts.labels[d.classId] ?? `class_${d.classId}`,
		};
	});
}

/* ── COCO 80 class labels ─────────────────────────────────────────────── */

export const COCO_LABELS: readonly string[] = [
	'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
	'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
	'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
	'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
	'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
	'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
	'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
	'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
	'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
	'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
	'toothbrush',
] as const;
