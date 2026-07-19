<script lang="ts">
	import { onMount } from 'svelte';
	import { Detector, classColor } from './lib/detect.svelte.ts';
	import type { Detection } from './lib/yolo26-core.ts';

	const det = new Detector();
	let mode = $state<'image' | 'camera'>('image');

	let imgEl = $state<HTMLImageElement | null>(null);
	let videoEl = $state<HTMLVideoElement | null>(null);
	let overlay = $state<HTMLCanvasElement | null>(null);
	let imageUrl = $state('');
	let imgTick = $state(0);
	let webgpu = $state(true);
	let cameraOn = $state(false);
	let stream: MediaStream | null = null;
	let rafId = 0;

	onMount(() => {
		webgpu = Detector.webgpuAvailable();
		det.start();
		return () => {
			det.dispose();
			stopCamera();
		};
	});

	/** Draw detection boxes onto the overlay, scaling natural→display px. */
	function drawBoxes(natW: number, natH: number, dispW: number, dispH: number, dets: Detection[]) {
		if (!overlay || dispW === 0 || dispH === 0) return;
		overlay.width = dispW;
		overlay.height = dispH;
		const sx = dispW / natW;
		const sy = dispH / natH;
		const ctx = overlay.getContext('2d')!;
		ctx.clearRect(0, 0, dispW, dispH);
		ctx.lineWidth = 2;
		ctx.font = '13px ui-sans-serif, system-ui';
		ctx.textBaseline = 'top';
		for (const d of dets) {
			const x = d.x1 * sx;
			const y = d.y1 * sy;
			const w = (d.x2 - d.x1) * sx;
			const h = (d.y2 - d.y1) * sy;
			const color = classColor(d.classId);
			ctx.strokeStyle = color;
			ctx.strokeRect(x, y, w, h);
			const tag = `${d.label} ${(d.score * 100).toFixed(0)}%`;
			const tw = ctx.measureText(tag).width + 8;
			ctx.fillStyle = color;
			ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
			ctx.fillStyle = '#000';
			ctx.fillText(tag, x + 4, Math.max(0, y - 17));
		}
	}

	// Static-image overlay: redraw when detections or the image change.
	$effect(() => {
		const dets = det.detections;
		imgTick;
		if (mode !== 'image' || !imgEl || !imgEl.complete || imgEl.naturalWidth === 0) return;
		drawBoxes(imgEl.naturalWidth, imgEl.naturalHeight, imgEl.clientWidth, imgEl.clientHeight, dets);
	});

	function onFile(e: Event) {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		if (imageUrl) URL.revokeObjectURL(imageUrl);
		imageUrl = URL.createObjectURL(file);
		det.detections = [];
	}

	async function runDetect() {
		if (!imgEl) return;
		await det.detect(imgEl);
	}

	/* ── Camera realtime ── */

	async function startCamera() {
		if (!det.loaded) return;
		try {
			stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
			if (!videoEl) return;
			videoEl.srcObject = stream;
			await videoEl.play();
			cameraOn = true;
			loop();
		} catch (err) {
			det.error = err instanceof Error ? err.message : String(err);
		}
	}

	function stopCamera() {
		cameraOn = false;
		if (rafId) cancelAnimationFrame(rafId);
		rafId = 0;
		stream?.getTracks().forEach((t) => t.stop());
		stream = null;
	}

	function loop() {
		if (!cameraOn || !videoEl) return;
		// Issue a new inference only when the worker is free (det.detect self-guards).
		if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
			det.detect(videoEl);
			drawBoxes(videoEl.videoWidth, videoEl.videoHeight, videoEl.clientWidth, videoEl.clientHeight, det.detections);
		}
		rafId = requestAnimationFrame(loop);
	}

	function switchMode(m: 'image' | 'camera') {
		if (m === mode) return;
		if (mode === 'camera') stopCamera();
		det.detections = [];
		mode = m;
	}
</script>

<main class="wrap">
	<header>
		<h1>眼 <span class="sub">manako · browser-local YOLO26</span></h1>
		<p class="tagline">On-device object detection — WebGPU/wasm, zero upload, zero server inference.</p>
	</header>

	{#if !webgpu}
		<div class="warn">WebGPU 非対応ブラウザです。wasm フォールバックで動作しますが低速になります（Chrome/Edge 113+ 推奨）。</div>
	{/if}

	<section class="controls">
		<label>
			Model
			<select bind:value={det.modelId} disabled={det.loading || cameraOn}>
				{#each det.models as m}
					<option value={m.id}>{m.label} (~{m.sizeMb}MB)</option>
				{/each}
			</select>
		</label>
		<button onclick={() => det.loadModel()} disabled={det.loading || det.loaded}>
			{det.loaded ? '✓ Loaded' : det.loading ? 'Loading…' : 'Load model'}
		</button>
		<label class="slider">conf {det.confThreshold.toFixed(2)}
			<input type="range" min="0.05" max="0.9" step="0.05" bind:value={det.confThreshold} />
		</label>
		<label class="slider">IoU {det.iouThreshold.toFixed(2)}
			<input type="range" min="0.1" max="0.9" step="0.05" bind:value={det.iouThreshold} />
		</label>
	</section>

	<section class="controls tabs">
		<button class:active={mode === 'image'} onclick={() => switchMode('image')}>🖼 Image</button>
		<button class:active={mode === 'camera'} onclick={() => switchMode('camera')}>📷 Camera</button>
		<span class="status">{det.statusLabel}</span>
	</section>

	<section class="controls">
		{#if mode === 'image'}
			<input type="file" accept="image/*" onchange={onFile} />
			<button onclick={runDetect} disabled={!det.loaded || !imageUrl}>Detect</button>
		{:else}
			{#if !cameraOn}
				<button onclick={startCamera} disabled={!det.loaded}>Start camera</button>
			{:else}
				<button onclick={stopCamera}>Stop camera</button>
			{/if}
			<span class="hint">映像はブラウザ内でのみ処理され、外部に送信されません。</span>
		{/if}
	</section>

	{#if det.loading}
		<div class="bar"><div class="fill" style="width:{det.progress}%"></div></div>
	{/if}
	{#if det.error}
		<div class="err">⚠ {det.error}</div>
	{/if}

	<section class="stage">
		{#if mode === 'image'}
			{#if imageUrl}
				<div class="canvas-wrap">
					<!-- svelte-ignore a11y_missing_attribute -->
					<img bind:this={imgEl} src={imageUrl} onload={() => (imgTick += 1)} />
					<canvas bind:this={overlay}></canvas>
				</div>
			{:else}
				<div class="placeholder">画像を選択してください（画像はブラウザ内で処理され、外部に送信されません）。</div>
			{/if}
		{:else}
			<div class="canvas-wrap">
				<!-- svelte-ignore a11y_media_has_caption -->
				<video bind:this={videoEl} playsinline muted></video>
				<canvas bind:this={overlay}></canvas>
			</div>
		{/if}
	</section>

	{#if det.detections.length}
		<section class="list">
			{#each det.detections as d}
				<span class="chip" style="border-color:{classColor(d.classId)}">{d.label} {(d.score * 100).toFixed(0)}%</span>
			{/each}
		</section>
	{/if}

	<footer>
		Apache-2.0 + etzhayyim Charter Rider. Model weights operator-supplied (Ultralytics YOLO26 = AGPL-3.0; not bundled).
		On-device only · no telemetry · object detection only (no face-ID / no surveillance).
	</footer>
</main>
