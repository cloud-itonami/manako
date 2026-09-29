# operator quickstart — manako（眼）

clone から「ブラウザで物体検出が動く」までの手順。**下の各段は 2026-08-14 に実際に踏んで、
出力をそのまま貼ってある**（node v26.3.0 / pnpm 10.26.2 / macOS arm64）。踏んでいない段は
そう明記してある —— 読んだだけの手順と、走らせた手順を混ぜない。

作業ディレクトリは全段とも:

```bash
cd appview/etzhayyim-wasm-manako-m4n4k0v1/svelte
```

---

## 1. 依存を入れる

```bash
pnpm install
```

実測:

```
dependencies:      + onnxruntime-web 1.26.0
devDependencies:   + svelte 5.56.1  + vite 6.4.3  + tsx 4.22.4  + typescript 5.9.3  …
Done in 1.2s using pnpm v10.26.2
```

**`Ignored build scripts: esbuild@0.25.12, esbuild@0.28.0, protobufjs@7.6.2` という警告が
出るが、これは止まる理由ではない。** 実測でこの警告が出たままテスト（§2）もビルド（§3）も
dev server（§4）も通っている。`pnpm approve-builds` は要らない。

## 2. 純関数コアのテストを通す（ここが唯一の緑判定）

```bash
pnpm test
```

実測 —— **12 pass / 0 fail**:

```
✔ computeLetterbox: landscape source into square input
✔ computeLetterbox: portrait source pads horizontally
✔ unletterboxBox: round-trips a source box through model space
✔ unletterboxBox: clamps to image bounds
✔ iou: identical boxes = 1, disjoint = 0, half-overlap
✔ nms: suppresses overlapping same-class, keeps separate + other class
✔ detectLayout: recognises chw / hwc / nms-free
✔ decodeRaw (chw): finds one box above threshold
✔ postprocess (raw chw): two overlapping anchors collapse to one detection
✔ postprocess (nms-free): passthrough thresholds + labels + maps back
✔ postprocess (nms-free transposed): handles (1, 6, N)
✔ COCO labels: exactly 80 classes, person first
ℹ tests 12   ℹ pass 12   ℹ fail 0
```

これが赤い状態で先へ進まない。**この 12 本が触るのは `src/lib/yolo26-core.ts` だけ**で、
worker・モデル取得・WebGPU は 1 行も通らない。緑は「デコード算術が正しい」であって
「アプリが動く」ではない。

## 3. ビルドする

```bash
pnpm build     # → ../_svelte
```

実測（`vite v6.4.3`、1.49s、112 modules）:

| 成果物 | サイズ |
|---|---|
| `../_svelte/index.html` | 0.44 kB |
| `assets/index-*.js` | 47.67 kB（gzip 18.85 kB） |
| `assets/index-*.css` | 7.05 kB |
| `assets/detect-worker-*.js` | 5.46 kB |
| `assets/ort.bundle.min-*.js` | 403.34 kB |
| `assets/ort-wasm-simd-threaded.jsep-*.wasm` | **26,239.91 kB**（gzip 6,154.99 kB） |

**26 MB の wasm は ONNX Runtime 本体であって、モデルではない。** モデルはこれとは別に、
実行時にダウンロードされる（§5）。配信の帯域を見積もるときはこの 2 つを足す。

> この workspace（superproject）の中でビルドするときは、repo-wide の resource governor を
> 通す —— 高負荷ビルドの同時実行は 1 本に制限されている:
> `node <superproject>/scripts/resource-guard.mjs run build -- pnpm build`
> 上の実測もこの経路で取った。standalone clone なら素の `pnpm build` でよい。

## 4. dev server を上げる

```bash
pnpm dev --port 5190
```

実測 —— **`127.0.0.1` では繋がらない。`localhost` を使う**:

```
curl http://127.0.0.1:5190/   →  Failed to connect (exit 7)
curl http://localhost:5190/   →  200
curl http://[::1]:5190/       →  200
```

vite は IPv6 の loopback にだけ bind する。`127.0.0.1` を叩いて「起動していない」と
誤診しない。200 が返れば `<title>眼 manako — Browser-local YOLO26 Object Detection</title>` が
見える。

## 5. モデルを置く —— ここが唯一の詰まりどころ

### 5a. なぜ CDN 経路が今日は使えないのか

`src/lib/models.ts` は 4 つのモデルを登録している。うち 3 つ（`yolo26n` / `yolo26s` /
`yolo26m`）は `https://cdn.etzhayyim.com/models/yolo26/…` を指す。

**2026-08-14 実測: `cdn.etzhayyim.com` は NXDOMAIN。** `manako.etzhayyim.com` も同じく
解決しない。

```
$ host cdn.etzhayyim.com
Host cdn.etzhayyim.com not found: 3(NXDOMAIN)
```

→ **今日この repo を clone した operator が取れる経路は、自己ホストの `yolo26n-local`
だけ**である。既定の `DEFAULT_MODEL_ID` も `yolo26n-local` なので、既定のまま進めばよい。

### 5b. `.onnx` を作る（**この段は未実走**）

**以下は踏んでいない。** Ultralytics は AGPL-3.0 で、この repo には vendor しない
operator 側ツールチェインなので、検証環境には入れていない（`NOTICE` の G2）。

```bash
pip install ultralytics                                        # AGPL — operator 側にのみ置く
yolo export model=yolo26n.pt format=onnx imgsz=640 nms=True    # NMS-free（推奨）
```

出来た artifact が満たすべき条件は、`src/lib/` を読めば分かる範囲で次のとおり:

| 項目 | 要求 |
|---|---|
| 入力 | 名前は `session.inputNames[0]` を採るので任意。形は `[1, 3, 640, 640]` NCHW float32、`/255` 正規化、pad は `rgb(114,114,114)` |
| 出力 | `(1, N, 6)` = `[x1,y1,x2,y2,score,cls]`（`nms=True`）**または** raw head `(1, 4+nc, A)` / `(1, A, 4+nc)` |
| クラス数 | 80（COCO） |

`detectLayout()` がどちらの形かを自動判定するので、`nms=False` で export しても動く
（§2 のテストが両方の経路を固定している）。

### 5c. 置く場所 —— 置いたら dev server を**再起動する**

```bash
mkdir -p public/models/yolo26
cp /path/to/yolo26n.onnx public/models/yolo26/yolo26n.onnx
```

`public/models/` と `*.onnx` は `.gitignore` 済み。**commit しない。**

**`public/` を作る前に dev server を起動していたら、必ず再起動する。** vite は
publicDir を起動時に解決するので、後から `public/` を作っても**その回のプロセスからは
永久に見えない**。実測（2026-08-14、同一ファイルに対する同一 URL）:

| dev server の起動タイミング | `/models/yolo26/yolo26n.onnx` |
|---|---|
| `public/` を作る**前**に起動 | `200`, `text/html`, **394 バイト**（= §6 の罠と同じ顔） |
| `public/` がある状態で起動 | `200`, **26,239,907 バイト** |

ファイルは両方の測定で同じ場所に同じサイズで置かれていた。**違いは起動順だけ。**
これを知らないと「置いたのに 394 が返る」→「export が壊れている」と誤診する。

### 5d. 置けたことを確かめる（**必ずやる**）

```bash
curl -sS -o /dev/null -w 'HTTP=%{http_code} type=%{content_type} bytes=%{size_download}\n' \
  http://localhost:5190/models/yolo26/yolo26n.onnx
```

- **正しい**: `bytes=` が実ファイルのサイズ（数 MB〜数十 MB）
- **置けていない / 再起動していない**: `type=text/html bytes=394`

**判定に使うのは `bytes` と「`text/html` でないこと」であって、`content-type` の中身では
ない。** 実測では正常時の `content-type` は**空**で返る（vite は `.onnx` を知らないので
`application/octet-stream` も付けない）。

### 5e. 置いたモデルは deploy バンドルに入る

`public/` は `pnpm build` で `../_svelte` にコピーされる。実測: 26 MB のファイルを 1 つ
置いた状態でビルドすると **`../_svelte` は 51 MB**（ORT の wasm 26 MB + モデル 26 MB）。

モデルを CDN から配りたい（バンドルに入れたくない）なら `public/` には置かず、
`src/lib/models.ts` に自分のホストを指す entry を足す。**自己ホスト経路を選ぶということは、
モデルを配信 artifact に同梱するということ**である。Workers Assets のサイズ上限を
先に確認すること。

## 6. モデルを置かずに起動すると何が起きるか（実測）

**404 は返らない。** dev server は未知のパスに index.html を返す（SPA フォールバック）ので、
モデルが無いとき `/models/yolo26/yolo26n.onnx` は **`200 text/html`, 394 バイト**になる:

```
$ curl -D - -o body.bin http://localhost:5190/models/yolo26/yolo26n.onnx
HTTP/1.1 200 OK
Content-Type: text/html
$ wc -c < body.bin
394
```

`detect-worker.ts` の `fetchModelFile()` が持つ番人は `if (!res.ok) throw` の 1 つだけで、
**200 なので発火しない。** その 394 バイトの HTML はそのままモデルとして
`InferenceSession.create()` に渡る。実測（`onnxruntime-web` の wasm EP に同じ 394 バイトを
食わせた）:

```
buffer bytes = 394
InferenceSession.create threw ->
  Can't create a session. ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model
  because protobuf parsing failed.
```

→ **「モデルが無い」は「protobuf が壊れている」という顔で出てくる。** この文字列を見たら、
まず §5d の `curl` を叩き、次に §5c の**起動順**を疑う。壊れた `.onnx` を疑う前に、そもそも `.onnx` が
配信されているかを見る。

### 一度踏むと reload では直らない（コード読解、ブラウザ未実測）

`fetchModelFile()` は fetch した bytes を **失敗する前に OPFS へ書く**。読み出し側は
`f.size > 0` なら再取得せずにそれを返す。したがって **394 バイトの HTML が「キャッシュ済み
モデル」として残り、後から本物の `.onnx` を置いても同じエラーが出続ける**はずである。

**この持続の部分だけは実測していない**（OPFS はブラウザの中にしか無く、上の検証は node 上で
行ったため）。§6 の 200・394 バイト・protobuf エラーは実測、OPFS の居座りは
`detect-worker.ts` の読解による推論である。踏んだ疑いがあるなら DevTools →
Application → Storage → *Clear site data* を先に一度やる。

## 7. 今日ここまでで確かめられること / 確かめられないこと

| | 状態 |
|---|---|
| 純関数コア（letterbox・NMS・レイアウト判定・両デコーダ） | **実測で緑**（12/12、§2） |
| production ビルドが通る | **実測で緑**（§3） |
| dev server が配信する | **実測で緑**（§4、`localhost` のみ） |
| モデル欠落時の失敗の出方 | **実測**（§6） |
| `public/` の起動順トラップ・配信バイト数・バンドル膨張 | **実測**（§5c / §5d / §5e、26 MB のバイナリを stand-in にして両方向を測った） |
| 実 `.onnx` を積んだ end-to-end 検出 | **未実測** — operator が §5b を踏むまで到達できない |
| Chrome/WebGPU での実行 | **未実測** — `AGENTS.md` の R0 ステータスが言うとおり |
