/**
 * models.ts — YOLO26 model registry for manako (眼) browser-local detection.
 *
 * **No weights are committed to this repo.** Ultralytics YOLO26 is AGPL-3.0; bundling
 * its weights would taint the Apache-2.0 + Charter-Rider tree (and violates the
 * no-git-lfs rule). Each entry points at an ONNX file the OPERATOR exports and hosts
 * (B2 CDN or IPFS), exactly like gazo's SD weights. Until an operator provisions the
 * CDN path, the app loads nothing and says so — honest R0.
 *
 * Export recipe (operator, offline, one-time):
 *   pip install ultralytics            # AGPL — operator tooling, never vendored here
 *   yolo export model=yolo26n.pt format=onnx imgsz=640 nms=True   # NMS-free (preferred)
 *   # or  nms=False for the raw head — manako auto-detects either layout.
 *
 * @module
 */

import { COCO_LABELS } from './yolo26-core.ts';

export interface Yolo26Model {
	id: string;
	label: string;
	/** Ultralytics scale variant. */
	variant: 'n' | 's' | 'm' | 'l' | 'x';
	/** Approx ONNX size in MB (FP32). */
	sizeMb: number;
	/** Square model input edge (px). */
	inputSize: number;
	/** Number of classes. */
	numClasses: number;
	/** Class label table. */
	labels: readonly string[];
	/** Full URL to the ONNX file (operator-hosted; may be split into parts). */
	onnxParts: string[];
}

const CDN = 'https://cdn.etzhayyim.com/models/yolo26';

/** COCO-pretrained YOLO26 detect variants. Sizes are approximate FP32 ONNX. */
export const YOLO26_MODELS: readonly Yolo26Model[] = [
	{
		// Dev/self-host: served from this app's own /public (gitignored .onnx). No CDN, no
		// external fetch — used for local verification and any operator self-hosting.
		id: 'yolo26n-local',
		label: 'YOLO26-nano (self-hosted /models)',
		variant: 'n',
		sizeMb: 10,
		inputSize: 640,
		numClasses: 80,
		labels: COCO_LABELS,
		onnxParts: ['/models/yolo26/yolo26n.onnx'],
	},
	{
		id: 'yolo26n',
		label: 'YOLO26-nano (COCO, fastest)',
		variant: 'n',
		sizeMb: 11,
		inputSize: 640,
		numClasses: 80,
		labels: COCO_LABELS,
		onnxParts: [`${CDN}/yolo26n.onnx`],
	},
	{
		id: 'yolo26s',
		label: 'YOLO26-small (COCO, balanced)',
		variant: 's',
		sizeMb: 38,
		inputSize: 640,
		numClasses: 80,
		labels: COCO_LABELS,
		onnxParts: [`${CDN}/yolo26s.onnx`],
	},
	{
		id: 'yolo26m',
		label: 'YOLO26-medium (COCO, accurate)',
		variant: 'm',
		sizeMb: 79,
		inputSize: 640,
		numClasses: 80,
		labels: COCO_LABELS,
		onnxParts: [`${CDN}/yolo26m.onnx`],
	},
] as const;

export const DEFAULT_MODEL_ID = 'yolo26n-local';
