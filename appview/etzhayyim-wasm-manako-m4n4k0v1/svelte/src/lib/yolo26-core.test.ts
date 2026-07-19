/**
 * yolo26-core.test.ts — Node-runnable unit tests for the pure YOLO26 core.
 *
 * Run:  node --import tsx --test src/lib/yolo26-core.test.ts
 *
 * These cover the GPU-independent logic — letterbox geometry, IoU/NMS, output-layout
 * auto-detection, and both decode paths — on synthetic tensors. This is the honest
 * verifiable surface; live WebGPU inference needs a real exported YOLO26 ONNX model.
 *
 * @module
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
	computeLetterbox,
	unletterboxBox,
	iou,
	nms,
	detectLayout,
	decodeRaw,
	postprocess,
	COCO_LABELS,
	type BBox,
} from './yolo26-core.ts';

const NC = 80;
const OPTS = { numClasses: NC, labels: COCO_LABELS } as const;

test('computeLetterbox: landscape source into square input', () => {
	const lb = computeLetterbox(1280, 720, 640, 640);
	assert.equal(lb.scale, 0.5); // min(640/1280, 640/720) = 0.5
	assert.equal(lb.padX, 0); // 1280*0.5 = 640 → fills width
	assert.equal(lb.padY, Math.floor((640 - 360) / 2)); // 140
});

test('computeLetterbox: portrait source pads horizontally', () => {
	const lb = computeLetterbox(720, 1280, 640, 640);
	assert.equal(lb.scale, 0.5);
	assert.equal(lb.padY, 0);
	assert.equal(lb.padX, 140);
});

test('unletterboxBox: round-trips a source box through model space', () => {
	const lb = computeLetterbox(1280, 720, 640, 640);
	// Source box → model space: src*scale + pad
	const src: BBox = [100, 200, 300, 400];
	const model: BBox = [
		src[0] * lb.scale + lb.padX,
		src[1] * lb.scale + lb.padY,
		src[2] * lb.scale + lb.padX,
		src[3] * lb.scale + lb.padY,
	];
	const back = unletterboxBox(model, lb);
	for (let i = 0; i < 4; i++) assert.ok(Math.abs(back[i] - src[i]) < 1e-6);
});

test('unletterboxBox: clamps to image bounds', () => {
	const lb = computeLetterbox(640, 480, 640, 640);
	const back = unletterboxBox([-50, -50, 100000, 100000], lb);
	assert.deepEqual(back, [0, 0, 640, 480]);
});

test('iou: identical boxes = 1, disjoint = 0, half-overlap', () => {
	const a: BBox = [0, 0, 10, 10];
	assert.equal(iou(a, a), 1);
	assert.equal(iou(a, [20, 20, 30, 30]), 0);
	// b shares right half of a: intersection 5x10=50, union 100+100-50=150
	assert.ok(Math.abs(iou(a, [5, 0, 15, 10]) - 50 / 150) < 1e-9);
});

test('nms: suppresses overlapping same-class, keeps separate + other class', () => {
	const boxes: BBox[] = [
		[0, 0, 10, 10], // A score .9 cls0
		[1, 1, 11, 11], // overlaps A, score .8 cls0 → suppressed
		[100, 100, 110, 110], // separate, score .7 cls0 → kept
		[1, 1, 11, 11], // overlaps A but cls1 → kept (class-aware)
	];
	const scores = [0.9, 0.8, 0.7, 0.85];
	const classes = [0, 0, 0, 1];
	const kept = nms(boxes, scores, classes, 0.45, 300).sort((a, b) => a - b);
	assert.deepEqual(kept, [0, 2, 3]);
});

test('detectLayout: recognises chw / hwc / nms-free', () => {
	assert.equal(detectLayout([1, 84, 8400], NC), 'chw');
	assert.equal(detectLayout([1, 8400, 84], NC), 'hwc');
	assert.equal(detectLayout([1, 300, 6], NC), 'nms-free');
	assert.equal(detectLayout([1, 6, 300], NC), 'nms-free');
});

test('decodeRaw (chw): finds one box above threshold', () => {
	// dims [1, 84, A] with A=2. Anchor 0 = strong "car"(id2), anchor 1 = noise.
	const A = 2;
	const feat = 4 + NC;
	const data = new Float32Array(feat * A);
	const setChw = (a: number, c: number, v: number) => (data[c * A + a] = v);
	// anchor 0: cx=320 cy=320 w=100 h=80, class 2 score 0.9
	setChw(0, 0, 320);
	setChw(0, 1, 320);
	setChw(0, 2, 100);
	setChw(0, 3, 80);
	setChw(0, 4 + 2, 0.9);
	// anchor 1: all class scores ~0.1 (below 0.25)
	for (let c = 0; c < NC; c++) setChw(1, 4 + c, 0.1);
	const cand = decodeRaw(data, [1, feat, A], 'chw', NC, 0.25);
	assert.equal(cand.boxes.length, 1);
	assert.equal(cand.classes[0], 2);
	assert.deepEqual(cand.boxes[0], [270, 280, 370, 360]); // cxcywh→xyxy
});

test('postprocess (raw chw): two overlapping anchors collapse to one detection', () => {
	const A = 2;
	const feat = 4 + NC;
	const data = new Float32Array(feat * A);
	const setChw = (a: number, c: number, v: number) => (data[c * A + a] = v);
	// Two near-identical "person"(id0) boxes → NMS keeps the higher-scoring one.
	for (const [a, s] of [[0, 0.9], [1, 0.8]] as const) {
		setChw(a, 0, 320 + a); // cx
		setChw(a, 1, 320);
		setChw(a, 2, 200);
		setChw(a, 3, 200);
		setChw(a, 4 + 0, s);
	}
	const lb = computeLetterbox(640, 640, 640, 640); // identity letterbox
	const dets = postprocess(data, [1, feat, A], lb, OPTS);
	assert.equal(dets.length, 1);
	assert.equal(dets[0].label, 'person');
	assert.ok(Math.abs(dets[0].score - 0.9) < 1e-6); // Float32 storage tolerance
});

test('postprocess (nms-free): passthrough thresholds + labels + maps back', () => {
	// dims [1, N, 6]: row = [x1,y1,x2,y2,score,cls]
	const rows = [
		[100, 100, 200, 200, 0.95, 2], // car, kept
		[10, 10, 20, 20, 0.10, 5], // below conf, dropped
	];
	const data = new Float32Array(rows.flat());
	const lb = computeLetterbox(640, 640, 640, 640);
	const dets = postprocess(data, [1, rows.length, 6], lb, OPTS);
	assert.equal(dets.length, 1);
	assert.equal(dets[0].label, 'car');
	assert.deepEqual([dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2], [100, 100, 200, 200]);
});

test('postprocess (nms-free transposed): handles (1, 6, N)', () => {
	// Column-major: feature axis first. Two dets.
	const n = 2;
	const cols: number[][] = [
		[100, 50], // x1
		[100, 50], // y1
		[200, 60], // x2
		[200, 60], // y2
		[0.95, 0.10], // score
		[2, 5], // cls
	];
	const data = new Float32Array(cols.flat());
	const lb = computeLetterbox(640, 640, 640, 640);
	const dets = postprocess(data, [1, 6, n], lb, OPTS);
	assert.equal(dets.length, 1);
	assert.equal(dets[0].label, 'car');
});

test('COCO labels: exactly 80 classes, person first', () => {
	assert.equal(COCO_LABELS.length, 80);
	assert.equal(COCO_LABELS[0], 'person');
	assert.equal(COCO_LABELS[79], 'toothbrush');
});
