/**
 * detect.svelte.ts — Main-thread controller (Svelte 5 runes) driving the detect worker.
 *
 * Owns the worker lifecycle, accepts image / video-frame input, transfers an
 * ImageBitmap to the worker, and exposes reactive detection state to the UI.
 *
 * @module
 */

import DetectWorker from './detect-worker.ts?worker';
import type { DetectWorkerMessage } from './detect-worker.ts';
import type { Detection } from './yolo26-core.ts';
import { YOLO26_MODELS, DEFAULT_MODEL_ID, type Yolo26Model } from './models.ts';

export class Detector {
	worker: Worker | null = null;

	ready = $state(false);
	loading = $state(false);
	loaded = $state(false);
	provider = $state('');
	statusLabel = $state('idle');
	progress = $state(0);
	error = $state('');

	detections = $state<Detection[]>([]);
	inferenceMs = $state(0);
	srcW = $state(0);
	srcH = $state(0);

	confThreshold = $state(0.25);
	iouThreshold = $state(0.45);
	modelId = $state(DEFAULT_MODEL_ID);

	private busy = false;

	get models(): readonly Yolo26Model[] {
		return YOLO26_MODELS;
	}
	get model(): Yolo26Model {
		return YOLO26_MODELS.find((m) => m.id === this.modelId) ?? YOLO26_MODELS[0];
	}

	/** True only on a WebGPU-capable, secure browsing context. */
	static webgpuAvailable(): boolean {
		return typeof navigator !== 'undefined' && 'gpu' in navigator;
	}

	start() {
		if (this.worker) return;
		this.worker = new DetectWorker();
		this.worker.onmessage = (ev: MessageEvent<DetectWorkerMessage>) => this.onMessage(ev.data);
		this.worker.onerror = (e) => (this.error = e.message);
	}

	async loadModel() {
		this.start();
		this.loading = true;
		this.loaded = false;
		this.error = '';
		const m = this.model;
		this.post({
			type: 'load',
			load: { onnxParts: m.onnxParts, inputSize: m.inputSize, numClasses: m.numClasses, labels: m.labels },
		});
	}

	/** Run detection on any canvas-drawable image source. */
	async detect(source: ImageBitmapSource) {
		if (!this.loaded || this.busy) return;
		this.busy = true;
		this.statusLabel = 'detecting…';
		const bitmap = await createImageBitmap(source);
		this.post({ type: 'detect', detect: { bitmap, confThreshold: this.confThreshold, iouThreshold: this.iouThreshold } }, [bitmap]);
	}

	private onMessage(m: DetectWorkerMessage) {
		switch (m.type) {
			case 'ready':
				this.ready = true;
				break;
			case 'progress':
				if (m.progress) {
					this.statusLabel = m.progress.label;
					this.progress = m.progress.pct;
				}
				break;
			case 'loaded':
				this.loading = false;
				this.loaded = true;
				this.provider = m.provider ?? '';
				this.statusLabel = `ready · ${this.provider.toUpperCase()}`;
				break;
			case 'result':
				if (m.result) {
					this.detections = m.result.detections;
					this.inferenceMs = m.result.inferenceMs;
					this.srcW = m.result.srcW;
					this.srcH = m.result.srcH;
					this.statusLabel = `${m.result.detections.length} objects · ${m.result.inferenceMs.toFixed(0)}ms`;
				}
				this.busy = false;
				break;
			case 'error':
				this.error = m.error ?? 'unknown error';
				this.loading = false;
				this.busy = false;
				this.statusLabel = 'error';
				break;
		}
	}

	private post(msg: DetectWorkerMessage, transfer?: Transferable[]) {
		this.worker?.postMessage(msg, transfer ?? []);
	}

	dispose() {
		this.worker?.terminate();
		this.worker = null;
	}
}

/** Deterministic per-class color for box drawing. */
export function classColor(classId: number): string {
	const hue = (classId * 47) % 360;
	return `hsl(${hue} 85% 55%)`;
}
