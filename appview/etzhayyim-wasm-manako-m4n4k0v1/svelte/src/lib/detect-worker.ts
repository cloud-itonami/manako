/**
 * detect-worker.ts — Dedicated Web Worker running YOLO26 inference via ONNX Runtime.
 *
 * Execution provider preference: **WebGPU → wasm fallback** (same edge carve-out as
 * gazo / ameno; ADR-2606012100 ameno node class, ADR-2605215000 browser-WebGPU).
 * 100% on-device — the image bytes never leave the browser. No network except the
 * one-time model download from the operator CDN (cached in OPFS).
 *
 * Pipeline: ImageBitmap → letterbox onto OffscreenCanvas → NCHW Float32 tensor →
 * session.run → `postprocess()` (pure core) → detections back to main thread.
 *
 * @module
 */

import {
	computeLetterbox,
	postprocess,
	type Detection,
	type LetterboxParams,
} from './yolo26-core.ts';

export type DetectMsgType = 'load' | 'detect' | 'ready' | 'loaded' | 'result' | 'progress' | 'error';

export interface DetectWorkerMessage {
	type: DetectMsgType;
	load?: {
		/** ONNX file URLs (concatenated if >1). */
		onnxParts: string[];
		inputSize: number;
		numClasses: number;
		labels: readonly string[];
	};
	detect?: {
		bitmap: ImageBitmap;
		confThreshold: number;
		iouThreshold: number;
	};
	result?: {
		detections: Detection[];
		inferenceMs: number;
		srcW: number;
		srcH: number;
	};
	progress?: { label: string; pct: number };
	/** Which execution provider actually bound (for 'loaded'). */
	provider?: string;
	error?: string;
}

/* ── ONNX Runtime lazy import ── */

let ort: typeof import('onnxruntime-web') | null = null;
async function getOrt() {
	if (!ort) {
		ort = await import('onnxruntime-web');
		ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
		ort.env.wasm.simd = true;
	}
	return ort;
}

/* ── Model state ── */

let session: import('onnxruntime-web').InferenceSession | null = null;
let inputName = 'images';
let cfg: NonNullable<DetectWorkerMessage['load']> | null = null;

/* ── OPFS-cached model fetch (mirrors gazo) ── */

async function opfsDir(parts: string[]): Promise<FileSystemDirectoryHandle> {
	let dir = await navigator.storage.getDirectory();
	for (const p of parts) dir = await dir.getDirectoryHandle(p, { create: true });
	return dir;
}

async function fetchModelFile(url: string, onPct?: (p: number) => void): Promise<ArrayBuffer> {
	// Resolve relative URLs (self-hosted /models/…) against the worker origin.
	const key = new URL(url, self.location.origin).pathname.replace(/^\//, '');
	const segs = key.split('/');
	const file = segs.pop()!;
	try {
		const dir = await opfsDir(segs);
		const fh = await dir.getFileHandle(file);
		const f = await fh.getFile();
		if (f.size > 0) {
			onPct?.(100);
			return await f.arrayBuffer();
		}
	} catch {
		/* miss */
	}
	const res = await fetch(new URL(url, self.location.origin).href);
	if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`);
	const total = Number(res.headers.get('content-length') || 0);
	const reader = res.body!.getReader();
	const chunks: Uint8Array[] = [];
	let got = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		got += value.byteLength;
		if (total) onPct?.(Math.round((got / total) * 100));
	}
	const buf = new Uint8Array(got);
	let off = 0;
	for (const c of chunks) {
		buf.set(c, off);
		off += c.byteLength;
	}
	try {
		const dir = await opfsDir(segs);
		const fh = await dir.getFileHandle(file, { create: true });
		const w = await fh.createWritable();
		await w.write(buf);
		await w.close();
	} catch {
		/* non-fatal */
	}
	return buf.buffer;
}

async function fetchParts(urls: string[], onPct?: (p: number) => void): Promise<ArrayBuffer> {
	if (urls.length === 1) return fetchModelFile(urls[0], onPct);
	const bufs: ArrayBuffer[] = [];
	let totalSize = 0;
	for (let i = 0; i < urls.length; i++) {
		const b = await fetchModelFile(urls[i], (p) => onPct?.(Math.round(((i + p / 100) / urls.length) * 100)));
		bufs.push(b);
		totalSize += b.byteLength;
	}
	const out = new Uint8Array(totalSize);
	let off = 0;
	for (const b of bufs) {
		out.set(new Uint8Array(b), off);
		off += b.byteLength;
	}
	return out.buffer;
}

/* ── Load ── */

async function load(spec: NonNullable<DetectWorkerMessage['load']>) {
	cfg = spec;
	post({ type: 'progress', progress: { label: 'Downloading YOLO26 model…', pct: 0 } });
	const buffer = await fetchParts(spec.onnxParts, (pct) =>
		post({ type: 'progress', progress: { label: `Downloading YOLO26 model… ${pct}%`, pct } }),
	);
	const ortLib = await getOrt();
	post({ type: 'progress', progress: { label: 'Initializing session…', pct: 100 } });
	let provider = 'webgpu';
	try {
		session = await ortLib.InferenceSession.create(buffer, { executionProviders: ['webgpu', 'wasm'] });
	} catch {
		provider = 'wasm';
		session = await ortLib.InferenceSession.create(buffer, { executionProviders: ['wasm'] });
	}
	inputName = session.inputNames[0] ?? 'images';
	post({ type: 'loaded', provider });
}

/* ── Preprocess: ImageBitmap → letterboxed NCHW Float32 ── */

function preprocess(
	bitmap: ImageBitmap,
	inputSize: number,
): { tensorData: Float32Array; lb: LetterboxParams } {
	const lb = computeLetterbox(bitmap.width, bitmap.height, inputSize, inputSize);
	const canvas = new OffscreenCanvas(inputSize, inputSize);
	const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
	// Neutral gray pad (Ultralytics uses 114/255).
	ctx.fillStyle = 'rgb(114,114,114)';
	ctx.fillRect(0, 0, inputSize, inputSize);
	const drawW = Math.round(bitmap.width * lb.scale);
	const drawH = Math.round(bitmap.height * lb.scale);
	ctx.drawImage(bitmap, lb.padX, lb.padY, drawW, drawH);
	const { data } = ctx.getImageData(0, 0, inputSize, inputSize); // RGBA HWC

	const plane = inputSize * inputSize;
	const out = new Float32Array(3 * plane); // NCHW, normalized /255
	for (let i = 0; i < plane; i++) {
		out[i] = data[i * 4] / 255; // R
		out[plane + i] = data[i * 4 + 1] / 255; // G
		out[2 * plane + i] = data[i * 4 + 2] / 255; // B
	}
	return { tensorData: out, lb };
}

/* ── Detect ── */

async function detect(req: NonNullable<DetectWorkerMessage['detect']>) {
	if (!session || !cfg) throw new Error('model not loaded');
	const ortLib = await getOrt();
	const { bitmap, confThreshold, iouThreshold } = req;
	const { tensorData, lb } = preprocess(bitmap, cfg.inputSize);
	const input = new ortLib.Tensor('float32', tensorData, [1, 3, cfg.inputSize, cfg.inputSize]);

	const t0 = performance.now();
	const out = await session.run({ [inputName]: input });
	const t1 = performance.now();

	const outName = session.outputNames[0];
	const tensor = out[outName];
	const detections = postprocess(tensor.data as Float32Array, tensor.dims as number[], lb, {
		numClasses: cfg.numClasses,
		labels: cfg.labels,
		confThreshold,
		iouThreshold,
	});

	post({
		type: 'result',
		result: { detections, inferenceMs: t1 - t0, srcW: bitmap.width, srcH: bitmap.height },
	});
	bitmap.close();
}

/* ── Plumbing ── */

function post(m: DetectWorkerMessage, transfer?: Transferable[]) {
	self.postMessage(m, { transfer: transfer ?? [] });
}

self.onmessage = async (ev: MessageEvent<DetectWorkerMessage>) => {
	const m = ev.data;
	try {
		if (m.type === 'load' && m.load) await load(m.load);
		else if (m.type === 'detect' && m.detect) await detect(m.detect);
	} catch (err) {
		post({ type: 'error', error: err instanceof Error ? err.message : String(err) });
	}
};

post({ type: 'ready' });
