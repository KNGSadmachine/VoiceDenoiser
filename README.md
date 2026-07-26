# VoiceDenoiser

[English](README_en.md)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

AI音声データセットのための一括ノイズ除去ツール

TTS・RVC・SoVITSなどの学習用データセットを入れると、AIモデルがホワイトノイズや環境音を自動で除去します。フォルダを指定して放置するだけで、面倒なデータセットのノイズ編集作業が不要になります。

![screenshot](docs/screenshot.png)

## 特徴

- **一括処理** — 入力フォルダを再帰的にスキャンし、フォルダ構造・ファイル名を保ったまま出力
- **AIノイズ除去** — 従来型のノイズゲートと違い、声の成分を学習したAIモデルが「声以外」を除去
- **3段階のエンジン** — 標準(DeepFilterNet=速い・安全)/強力(Resemble Enhance=リップノイズ等にも効く)/最強(修復あり)を試聴で聴き比べて選択
- **学習データを壊さない** — サンプリングレートは元ファイルを維持。除去強度も調整可能
- **試聴機能** — 一括実行の前に1ファイルだけ変換して処理前後を聴き比べ
- **バイノーラル保持モード** — L/Rを混ぜずに左右別々にノイズ除去し、ステレオのまま出力
- **後処理** — ノーマライズ(-3dB)、前後の無音カットをオプションで実行
- **レジューム対応** — 中断しても処理済みファイルはスキップして再開
- **GUI** — Gradio製。ブラウザで操作、GPUがあれば自動で使用(CPUでも動作)

## 動作環境

- Python 3.10 または 3.11
- Windows / Linux / macOS (Apple SiliconではCPU動作)
- NVIDIA GPU 推奨(なくても動きますが、特に強力/最強エンジンはCPUだとかなり遅くなります)

## セットアップ

```bash
git clone https://github.com/ReineHonoka/VoiceDenoiser.git
cd VoiceDenoiser
setup.bat        # Windows
# ./setup.sh     # Linux / macOS
```

venvの作成、PyTorch(GPUを自動判定してCUDA版/CPU版を選択)、依存パッケージのインストール、AIモデルのダウンロード(約700MB)まで自動で行います。

モデルのダウンロードを後回しにする場合は `SKIP_MODEL_DOWNLOAD=1 ./setup.sh` としてください。アプリ本体は起動でき、モデルは必要時にダウンロードされます。

## 使い方

```bash
run.bat          # Windows
# ./run.sh       # Linux
```

ブラウザで `http://127.0.0.1:7860` が開きます。

1. 音声ファイルを `dataset/raw/` に入れる(GUIに直接ドロップ、または別フォルダのパス指定でもOK)
2. バイノーラル音源は「処理モード」で「バイノーラル保持モード（LRを別々に処理）」を選ぶ
3. 試聴で処理前後を聴き比べて、エンジンと強度を選ぶ
4. 「一括処理開始」を押して放置 → `dataset/clean/` に出力

### バイノーラル音源について

通常モードでは、Resemble Enhanceが左右を平均してモノラル化してから処理します。左右の空間情報を残したい場合は、GUIの「処理モード」で「バイノーラル保持モード（LRを別々に処理）」を選択してください。このモードではL/Rを混ぜず、それぞれを独立してノイズ除去して2chのまま保存します。DeepFilterNetはこのモードでも左右を別チャンネルとして処理します。

### エンジンの選び方

| エンジン | 速度 | 向いているノイズ | 備考 |
|---|---|---|---|
| 標準 (DeepFilterNet) | 速い | ホワイトノイズ・環境音 | 声質が変わりにくい。まずはこれ |
| 強力 (Resemble Enhance) | 遅い | リップノイズ等の突発音 | 標準で取れない場合に |
| 最強 (Resemble Enhance 修復あり) | 遅い | 上記+音質補正 | 声質が変わるリスクあり。試聴必須 |

## 対応フォーマット

wav / flac / mp3 / ogg(非可逆のmp3/oggはwavで出力されます)

## License

[MIT License](LICENSE)

## Credits

- ノイズ除去モデル: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT / Apache-2.0)
- ノイズ除去モデル: [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) — `resemble_enhance/` に推論部のみ同梱(deepspeed依存を除去してWindows対応)
- UIテーマ: [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku) (Apache-2.0)
