# manako（眼）— browser-local object detection

**`manako` は「眼（まなこ）」で、機能を名前が示さない repo なのでここで名乗る。**
画像・映像フレームの中の物体を **ブラウザの中だけで** 検出する appview である。
YOLO26 の ONNX モデルを ONNX Runtime Web（WebGPU → wasm フォールバック）で走らせ、
COCO 80 クラスの矩形を返す。**画像はブラウザから出ない** — サーバ推論も、アップロードも、
テレメトリも無い。

- **中身**: `appview/etzhayyim-wasm-manako-m4n4k0v1/svelte/`（Svelte 5 CSR + Vite）
- **判断の核**: `src/lib/yolo26-core.ts` — GPU も DOM も使わない純関数
  （letterbox / un-letterbox / IoU / class-aware NMS / 出力レイアウト判定 / デコード）。
  ここだけが単体テストの対象で、12 本ある。
- **機構**: `src/lib/detect-worker.ts`（Web Worker、モデル取得と `session.run`）と
  `src/lib/detect.svelte.ts`（メインスレッド側の受け渡し）。

## 隣の repo との境界

同じ「エッジ推論の切り出し」に属する 3 本のうち、**manako は視覚の検出**を持つ:

| repo | 何をブラウザで走らせるか |
|---|---|
| `gazo` | 画像生成（Stable Diffusion） |
| `ameno` | 言語モデル |
| **`manako`** | **物体検出（YOLO26 / COCO-80）** |

manako は**顔認識でも人物再同定でも追跡でもない**。検出するのは COCO のクラスであって
個人ではない（`AGENTS.md` の G3）。

## 動かす

**operator が最初に踏む手順は `docs/operator-quickstart.md`。** 実際に踏んで、各段の
実測出力（テスト 12/12、build のバンドル内訳、dev server の応答コード）をそこに書いてある。
**モデルを置かずに起動したときに何が起きるか**という、コードを読んだだけでは見えない罠も
そこに実測付きで書いた。

最短だけ再掲する:

```bash
cd appview/etzhayyim-wasm-manako-m4n4k0v1/svelte
pnpm install
pnpm test     # → 12 pass / 0 fail
```

## モデルは、この repo に無い

Ultralytics YOLO26 の重みと export ツールチェインは **AGPL-3.0** で、この木
（Apache-2.0 + Charter Rider）には vendor しない。`.onnx` は operator が自分で export して
自分で配る（`src/lib/models.ts` のレジストリ）。`.gitignore` が `public/models/` と
`*.onnx` / `*.pt` を落としているのはそのため。詳細は `NOTICE`。

**2026-08-14 実測: レジストリが指す `cdn.etzhayyim.com` は NXDOMAIN で、`manako.etzhayyim.com`
も同じく解決しない。** つまり今日この repo を clone した operator が使える経路は
**自己ホスト（`yolo26n-local`）だけ**である。手順は quickstart に書いた。

## もっと深いところ

設計・実証履歴・不変条件（G1〜G5）・アーキテクチャ図は **`AGENTS.md`** が正本。
この README はそれを繰り返さない（同じ主張を 2 箇所に置くと、片方だけが直る）。
