# VoiceDenoiser

[English](README_en.md)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

バイノーラル音声の定位・距離感・左右差を守る自動編集ツールです。

現在は **Phase 1: De-plosive → Mouth De-click** を実装済みです。録音直後の2ch素材から、破裂音と短い口クリックを検出して局所修復し、個別修復へ渡す32-bit float WAVと処理レポートを一括生成します。

```text
素材
  ↓
Phase 1: De-plosive → Mouth De-click  ← 実装済み
  ↓
個別修復（人手）
  ↓
Phase 2: ノイズリダクション → EQ → DeEsser
  ↓
手動の音量調整 → ラウドネス一致
```

今後の実装順と品質基準は [roadmap.md](roadmap.md) を参照してください。

## Phase 1の特徴

- **バイノーラル2ch専用** — L/Rを混ぜず、片耳側だけのイベントもそのチャンネルだけ修復
- **順序を固定** — 必ずDe-plosiveの後にMouth De-clickを実行
- **局所処理** — 破裂音にはイベント区間だけの低域減衰、口クリックにはサンプル長を変えない短時間補間を適用
- **保守的な自動判定** — 強い候補だけを自動修復し、曖昧な候補は音を変えず要確認として記録
- **空間と時間を維持** — サンプルレート、2ch、サンプル数、L/Rの並びを維持
- **中間マスター** — 複数工程での量子化劣化を避けるため32-bit float WAVで出力
- **処理レポート** — イベント時刻、対象チャンネル、信頼度、補正量、処理前後のpeak/RMSをJSONへ保存
- **安全なレジューム** — 入力内容・設定・処理バージョンが一致した成果物だけをスキップ
- **原子的書き出し** — 中断途中のファイルを完成品として残さない
- **試聴と一括処理** — 1ファイルの比較後、フォルダ構造を保ってまとめて処理

Phase 1では、ノーマライズ、無音カット、ノイズリダクション、EQ、DeEsserを行いません。

## 動作環境

- Python 3.10 または 3.11
- Windows / Linux / macOS
- Phase 1はCPUだけで動作
- Legacyノイズ除去ではNVIDIA GPUを推奨

## セットアップ

```bash
git clone https://github.com/ReineHonoka/VoiceDenoiser.git
cd VoiceDenoiser
setup.bat        # Windows
# ./setup.sh     # Linux / macOS
```

venvの作成、PyTorch、依存パッケージ、Legacy用AIモデルのダウンロードまで自動で行います。

モデルのダウンロードを後回しにする場合は `SKIP_MODEL_DOWNLOAD=1 ./setup.sh` としてください。Phase 1自体はAIモデルを使用しません。

## 使い方

```bash
run.bat          # Windows
# ./run.sh       # Linux / macOS
```

ブラウザで `http://127.0.0.1:7860` が開きます。

1. 2ch音声を `dataset/raw/` に入れる。GUIへの直接ドロップや別フォルダの指定も可能
2. 「処理フェーズ」で既定の「Phase 1 — De-plosive → Mouth De-click」を選ぶ
3. 「試聴用に1ファイル変換」で処理前後を確認
4. 「一括処理開始」を押す
5. `dataset/phase1/` のWAVを個別修復工程へ渡す

入力フォルダの構造とファイル名は出力側でも維持します。入力形式にかかわらず、Phase 1の出力拡張子は `.wav` です。

### レポート

各出力の隣に同じベース名の `.phase1.json` を保存します。

```text
dataset/phase1/scene/take.wav
dataset/phase1/scene/take.phase1.json
```

主な集計値:

- `de_plosive`: 自動修復した破裂音
- `mouth_de_click`: 自動修復した口クリック
- `review`: 自動変更せず、人間の確認へ回した候補

各イベントにはL/R、開始・終了時刻、信頼度、処理内容を記録します。

## 入力条件

- Phase 1は2ch音声のみ対応
- 16kHz以上
- 50ms以上
- wav / flac / mp3 / ogg

monoや3ch以上を自動変換すると意図しない定位変化につながるため、Phase 1では変換せずファイル単位のエラーにします。バッチ処理は他の正常なファイルを続行します。

## Legacyノイズ除去

従来のDeepFilterNet / Resemble Enhanceによるノイズ除去は「Legacy — 従来のノイズ除去」として残しています。画面の「Legacyノイズ除去の設定」からエンジン、強度、処理モード、後処理を選択できます。

LegacyはPhase 1とは別の処理です。Phase 1の出力へ自動的にLegacy処理を続けて適用することはありません。

## テスト

```bash
./venv/bin/python -m unittest discover -s tests -v
```

合成した破裂音・口クリックの検出修復、処理順、2ch・サンプル数の保持、反対チャンネルの非変更、32-bit float WAV、sidecar、レジューム判定を検証します。

## License

[MIT License](LICENSE)

## Credits

- Legacyノイズ除去モデル: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT / Apache-2.0)
- Legacyノイズ除去モデル: [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) — `resemble_enhance/` に推論部のみ同梱
- UIテーマ: [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku) (Apache-2.0)
