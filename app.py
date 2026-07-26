"""VoiceDenoiser — AI音声データセット一括ノイズ除去ツール"""
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch
import torchaudio

BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = BASE_DIR / "dataset" / "raw"
DEFAULT_OUTPUT = BASE_DIR / "dataset" / "clean"

SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# 可逆フォーマットは元の形式のまま、非可逆(mp3/ogg)はwavで出力する
LOSSY_EXTS = {".mp3", ".ogg"}

ENGINES = {
    "標準 (DeepFilterNet) — 速い・声質が変わりにくい。ホワイトノイズ向け": "dfn",
    "強力 (Resemble Enhance) — リップノイズ等の突発音にも効く": "re_denoise",
    "最強 (Resemble Enhance 修復あり) — 音質補正まで行う。声質が変わるリスクあり": "re_enhance",
}
DEFAULT_ENGINE = next(iter(ENGINES))

PROCESSING_MODES = {
    "通常モード（既存）": False,
    "バイノーラル保持モード（LRを別々に処理）": True,
}
DEFAULT_PROCESSING_MODE = next(iter(PROCESSING_MODES))

NORMALIZE_LABEL = "ノーマライズ (-3dB)"
TRIM_LABEL = "前後の無音カット"
POST_OPS = [NORMALIZE_LABEL, TRIM_LABEL]

NORMALIZE_PEAK_DB = -3.0
TRIM_THRESHOLD_DB = -40.0
TRIM_KEEP_SEC = 0.5

_model = None
_df_state = None


def get_model():
    global _model, _df_state
    if _model is None:
        from df.enhance import init_df
        _model, _df_state, _ = init_df()
    return _model, _df_state


def _denoise_dfn(audio: torch.Tensor, sr: int, strength_db: float) -> torch.Tensor:
    from df.enhance import enhance

    model, df_state = get_model()
    model_sr = df_state.sr()
    if sr != model_sr:
        audio = torchaudio.functional.resample(audio, sr, model_sr)

    # strength_db=100 で最大除去。それ未満は除去量の上限(dB)として渡す
    atten_lim = None if strength_db >= 100 else strength_db
    with torch.no_grad():
        cleaned = enhance(model, df_state, audio, atten_lim_db=atten_lim)

    if sr != model_sr:
        cleaned = torchaudio.functional.resample(cleaned, model_sr, sr)
    return cleaned


def _denoise_resemble_channel(audio: torch.Tensor, sr: int, do_enhance: bool) -> torch.Tensor:
    from resemble_enhance.enhancer.inference import denoise, enhance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = enhance if do_enhance else denoise
    cleaned, out_sr = fn(audio, sr, device)
    cleaned = cleaned.cpu()
    if out_sr != sr:
        cleaned = torchaudio.functional.resample(cleaned.unsqueeze(0), out_sr, sr).squeeze(0)
    return cleaned


def _denoise_resemble(audio: torch.Tensor, sr: int, do_enhance: bool,
                       preserve_stereo: bool = False) -> torch.Tensor:
    """Resemble Enhanceは1chずつ処理し、必要ならチャンネル数を保持する。"""
    if preserve_stereo and audio.shape[0] > 1:
        # Resemble Enhance自体はモノラル入力のみ対応するため、L/Rを混ぜずに
        # 同じ処理を各チャンネルへ適用する。これで空間情報を平均化しない。
        cleaned_channels = [
            _denoise_resemble_channel(channel, sr, do_enhance)
            for channel in audio
        ]
        lengths = {channel.shape[-1] for channel in cleaned_channels}
        if len(lengths) != 1:
            raise RuntimeError(
                "Resemble Enhanceの左右チャンネルで出力長が一致しません: "
                f"{sorted(lengths)}"
            )
        return torch.stack(cleaned_channels, dim=0)

    # 既存モードの挙動。Resemble Enhanceは左右を平均してモノラル化する。
    mono = audio.mean(0)
    return _denoise_resemble_channel(mono, sr, do_enhance).unsqueeze(0)


def _normalize(audio: torch.Tensor, peak_db: float = NORMALIZE_PEAK_DB) -> torch.Tensor:
    peak = audio.abs().max()
    if peak == 0:
        return audio
    return audio * (10 ** (peak_db / 20) / peak)


def _trim_silence(audio: torch.Tensor, sr: int,
                  threshold_db: float = TRIM_THRESHOLD_DB,
                  keep_sec: float = TRIM_KEEP_SEC) -> torch.Tensor:
    """先頭・末尾の無音を keep_sec 秒だけ残してカットする。中間の無音は触らない"""
    amp = audio.abs().amax(dim=0)
    voiced = (amp > 10 ** (threshold_db / 20)).nonzero()
    if len(voiced) == 0:
        return audio
    keep = int(keep_sec * sr)
    start = max(0, voiced[0].item() - keep)
    end = min(audio.shape[1], voiced[-1].item() + 1 + keep)
    return audio[:, start:end]


def denoise_file(in_path: Path, out_path: Path, strength_db: float,
                 engine: str = "dfn", post_ops: list | None = None,
                 preserve_stereo: bool = False):
    audio, sr = torchaudio.load(str(in_path))
    if engine == "dfn":
        cleaned = _denoise_dfn(audio, sr, strength_db)
    elif engine == "re_denoise":
        cleaned = _denoise_resemble(
            audio, sr, do_enhance=False, preserve_stereo=preserve_stereo
        )
    elif engine == "re_enhance":
        cleaned = _denoise_resemble(
            audio, sr, do_enhance=True, preserve_stereo=preserve_stereo
        )
    else:
        raise ValueError(f"unknown engine: {engine}")

    post_ops = post_ops or []
    if NORMALIZE_LABEL in post_ops:
        cleaned = _normalize(cleaned)
    if TRIM_LABEL in post_ops:
        cleaned = _trim_silence(cleaned, sr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), cleaned.T.numpy(), sr)


def collect_files(input_dir: Path, exclude_dir: Path | None = None):
    """input_dir 以下の対応音声を集める。出力フォルダが入力の中にある場合は除外する"""
    exclude = exclude_dir.resolve() if exclude_dir else None
    files = []
    for p in input_dir.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS):
            continue
        if exclude and exclude in p.resolve().parents:
            continue
        files.append(p)
    return sorted(files)


def output_path_for(in_file: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = in_file.relative_to(input_dir)
    if in_file.suffix.lower() in LOSSY_EXTS:
        rel = rel.with_suffix(".wav")
    return output_dir / rel


def validate_dirs(input_dir: str, output_dir: str):
    if not input_dir or not output_dir:
        raise gr.Error("入力フォルダと出力フォルダを指定してください。")
    in_p, out_p = Path(input_dir), Path(output_dir)
    if not in_p.is_dir():
        raise gr.Error(f"入力フォルダが見つかりません: {input_dir}")
    if in_p.resolve() == out_p.resolve():
        raise gr.Error("入力フォルダと出力フォルダは別にしてください。")
    return in_p, out_p


def resolve_jobs(mode: str, dropped: list | None, input_dir: str, output_dir: str):
    """選択中のタブに応じて処理対象の (入力ファイル, 出力先) を返す。
    ドロップモードは出力フォルダ直下にフラットに出力する"""
    if mode == "drop":
        if not dropped:
            raise gr.Error("音声ファイルをドロップしてください。")
        if not output_dir:
            raise gr.Error("出力フォルダを指定してください。")
        out_p = Path(output_dir)
        srcs = sorted(p for p in map(Path, dropped)
                      if p.suffix.lower() in SUPPORTED_EXTS)
        if not srcs:
            if any(not Path(p).suffix for p in dropped):
                raise gr.Error(
                    "フォルダはドロップできません。フォルダ内のファイルを全選択(Ctrl+A)して"
                    "ドロップするか、「フォルダ指定」タブでパスを指定してください。"
                )
            names = ", ".join(Path(p).name for p in dropped[:5])
            raise gr.Error(
                "対応形式 (wav/flac/mp3/ogg) のファイルがドロップされていません。"
                f"受け取ったファイル: {names}"
            )
        return [(s, output_path_for(s, s.parent, out_p)) for s in srcs]

    in_p, out_p = validate_dirs(input_dir, output_dir)
    files = collect_files(in_p, exclude_dir=out_p)
    if not files:
        raise gr.Error("入力フォルダに音声ファイルが見つかりません。")
    return [(f, output_path_for(f, in_p, out_p)) for f in files]


def preview(mode: str, dropped: list | None, input_dir: str, output_dir: str,
            strength_db: float, engine_label: str, processing_mode: str,
            post_ops: list):
    """最初の1ファイルだけ処理して聴き比べ用に返す"""
    src, _ = resolve_jobs(mode, dropped, input_dir, output_dir)[0]
    tmp = Path(tempfile.gettempdir()) / "voicedenoiser_preview.wav"
    denoise_file(
        src, tmp, strength_db, ENGINES[engine_label], post_ops,
        preserve_stereo=PROCESSING_MODES[processing_mode],
    )
    return str(src), str(tmp), f"試聴ファイル: {src.name}"


def run_batch(mode: str, dropped: list | None, input_dir: str, output_dir: str,
              strength_db: float, engine_label: str, processing_mode: str,
              post_ops: list,
              progress=gr.Progress()):
    jobs = resolve_jobs(mode, dropped, input_dir, output_dir)
    engine = ENGINES[engine_label]

    done = skipped = failed = 0
    logs = []
    if engine == "dfn":
        get_model()  # 進捗バーを出す前にモデルロードを済ませる

    for src, out_file in progress.tqdm(jobs, desc="ノイズ除去中"):
        if out_file.exists():
            skipped += 1
            continue
        try:
            denoise_file(
                src, out_file, strength_db, engine, post_ops,
                preserve_stereo=PROCESSING_MODES[processing_mode],
            )
            done += 1
        except Exception:
            failed += 1
            logs.append(f"失敗: {src.name}")
            logs.append(traceback.format_exc().splitlines()[-1])

    summary = (f"完了: {done}件 / スキップ(処理済み): {skipped}件 / "
               f"失敗: {failed}件 (全{len(jobs)}件)")
    return summary + ("\n" + "\n".join(logs) if logs else "")


def load_theme():
    # miku テーマ (NoCrypt/miku, Apache-2.0)。取得できない場合は標準テーマで起動
    try:
        return gr.Theme.from_hub("NoCrypt/miku")
    except Exception:
        return None


def build_ui():
    with gr.Blocks(title="VoiceDenoiser", theme=load_theme()) as demo:
        gr.Markdown(
            "# VoiceDenoiser\n"
            "AI音声データセットの一括ノイズ除去ツール。"
            "フォルダを指定して放置するだけで、全ファイルのノイズを除去します。\n\n"
            f"デバイス: **{'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}**"
        )
        with gr.Row(equal_height=False):
            # 左列: 設定
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 設定")
                    mode = gr.State("drop")
                    with gr.Tabs():
                        with gr.Tab("ファイルをドロップ") as tab_drop:
                            dropped = gr.File(
                                label="ここに音声ファイルをドロップ(複数可)",
                                file_count="multiple", type="filepath",
                            )
                        with gr.Tab("フォルダ指定") as tab_folder:
                            gr.Markdown(
                                "指定フォルダ内の全音声を、フォルダ構造を保ったまま処理します。"
                                "大量のファイルはこちらが高速です。"
                            )
                            input_dir = gr.Textbox(label="入力フォルダ", value=str(DEFAULT_INPUT))
                    tab_drop.select(lambda: "drop", None, mode)
                    tab_folder.select(lambda: "folder", None, mode)
                    output_dir = gr.Textbox(label="出力フォルダ", value=str(DEFAULT_OUTPUT))
                    engine = gr.Dropdown(
                        choices=list(ENGINES), value=DEFAULT_ENGINE, label="エンジン",
                        info="試聴で聴き比べて選んでください。強力/最強は処理が遅くなります",
                    )
                    processing_mode = gr.Radio(
                        choices=list(PROCESSING_MODES), value=DEFAULT_PROCESSING_MODE,
                        label="処理モード",
                        info=(
                            "バイノーラル保持モードではL/Rを混ぜずに別々に処理します。"
                            "Resemble Enhanceでも2chのまま出力します。"
                        ),
                    )
                    strength = gr.Slider(
                        10, 100, value=100, step=5, label="ノイズ除去強度 (dB)",
                        info="標準エンジンのみ有効。100=最大除去。声がこもる場合は下げてください",
                    )
                    post_ops = gr.CheckboxGroup(
                        choices=POST_OPS, value=[], label="後処理",
                        info="ノイズ除去のあとに実行。無音カットは前後の無音を0.5秒だけ残して詰めます",
                    )
            # 右列: 試聴 → 一括処理 → 結果
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 実行")
                    with gr.Row():
                        preview_btn = gr.Button("試聴用に1ファイル変換")
                        run_btn = gr.Button("一括処理開始", variant="primary")
                    preview_info = gr.Markdown()
                    with gr.Row():
                        audio_before = gr.Audio(label="処理前", type="filepath")
                        audio_after = gr.Audio(label="処理後", type="filepath")
                with gr.Group():
                    gr.Markdown("### 結果")
                    result = gr.Textbox(label="結果", lines=10, show_label=False)

        inputs = [mode, dropped, input_dir, output_dir, strength, engine, processing_mode, post_ops]
        preview_btn.click(preview, inputs, [audio_before, audio_after, preview_info])
        run_btn.click(run_batch, inputs, [result])
    return demo


if __name__ == "__main__":
    DEFAULT_INPUT.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    build_ui().launch(inbrowser=True)
