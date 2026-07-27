"""Binaural-safe Phase 1 processing: De-plosive -> Mouth De-click."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt


PHASE1_VERSION = "1.0.0"
PHASE1_STEPS = ("de_plosive", "mouth_de_click")
CHANNEL_NAMES = ("L", "R")


class Phase1Error(ValueError):
    """Raised when an input cannot be processed safely as binaural audio."""


@dataclass(frozen=True)
class Phase1Config:
    """Conservative defaults for close-mic binaural voice."""

    analysis_chunk_sec: float = 8.0

    plosive_low_hz: float = 170.0
    plosive_mid_low_hz: float = 280.0
    plosive_mid_high_hz: float = 2400.0
    plosive_frame_ms: float = 24.0
    plosive_hop_ms: float = 5.0
    plosive_baseline_sec: float = 1.0
    plosive_candidate_excess_db: float = 9.0
    plosive_repair_excess_db: float = 12.0
    plosive_candidate_ratio_db: float = 3.0
    plosive_repair_ratio_db: float = 5.0
    plosive_min_level_dbfs: float = -42.0
    plosive_max_duration_ms: float = 320.0
    plosive_max_attenuation_db: float = 10.0
    plosive_max_lr_delta_db: float = 3.0

    click_highpass_hz: float = 1800.0
    click_baseline_ms: float = 20.0
    click_candidate_ratio: float = 18.0
    click_repair_ratio: float = 28.0
    click_min_level_dbfs: float = -55.0
    click_max_repair_ms: float = 1.5
    click_max_review_ms: float = 4.0

    @classmethod
    def conservative(cls) -> "Phase1Config":
        return cls()


@dataclass(frozen=True)
class Phase1Event:
    step: str
    channel: int
    start_sample: int
    end_sample: int
    confidence: float
    action: str
    score: float
    attenuation_db: float = 0.0

    def to_dict(self, sample_rate: int) -> dict[str, Any]:
        return {
            "step": self.step,
            "channel": CHANNEL_NAMES[self.channel],
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_sec": round(self.start_sample / sample_rate, 6),
            "end_sec": round(self.end_sample / sample_rate, 6),
            "confidence": round(self.confidence, 4),
            "action": self.action,
            "score": round(self.score, 4),
            "attenuation_db": round(self.attenuation_db, 2),
        }


def sidecar_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".phase1.json")


def _safe_sosfiltfilt(sos: np.ndarray, samples: np.ndarray) -> np.ndarray:
    if samples.size < 32:
        return sosfilt(sos, samples)
    return sosfiltfilt(sos, samples)


def _frame_rms(
    samples: np.ndarray, frame_size: int, hop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    frame_size = max(1, min(frame_size, samples.size))
    hop_size = max(1, hop_size)
    starts = np.arange(0, max(1, samples.size - frame_size + 1), hop_size)
    if starts.size == 0:
        starts = np.array([0])
    ends = np.minimum(starts + frame_size, samples.size)
    energy = np.concatenate(([0.0], np.cumsum(samples * samples, dtype=np.float64)))
    sums = energy[ends] - energy[starts]
    rms = np.sqrt(sums / np.maximum(1, ends - starts))
    centers = starts + (ends - starts) // 2
    return rms, centers


def _mask_spans(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    first = previous = int(indexes[0])
    for raw_index in indexes[1:]:
        index = int(raw_index)
        if index - previous > max_gap + 1:
            spans.append((first, previous + 1))
            first = index
        previous = index
    spans.append((first, previous + 1))
    return spans


def _deduplicate_events(events: Iterable[Phase1Event]) -> list[Phase1Event]:
    result: list[Phase1Event] = []
    ordered = sorted(events, key=lambda item: (item.channel, item.start_sample))
    for event in ordered:
        if (
            result
            and result[-1].channel == event.channel
            and result[-1].step == event.step
            and event.start_sample <= result[-1].end_sample
        ):
            previous = result.pop()
            preferred = event if event.confidence > previous.confidence else previous
            result.append(
                Phase1Event(
                    step=preferred.step,
                    channel=preferred.channel,
                    start_sample=min(previous.start_sample, event.start_sample),
                    end_sample=max(previous.end_sample, event.end_sample),
                    confidence=max(previous.confidence, event.confidence),
                    action=(
                        "repaired"
                        if "repaired" in (previous.action, event.action)
                        else "review"
                    ),
                    score=max(previous.score, event.score),
                    attenuation_db=max(
                        previous.attenuation_db, event.attenuation_db
                    ),
                )
            )
        else:
            result.append(event)
    return result


def _link_stereo_events(
    events: Iterable[Phase1Event],
    max_gap_samples: int,
    max_linked_duration_samples: int | None = None,
    max_attenuation_delta_db: float | None = None,
) -> list[Phase1Event]:
    """Give matching L/R detections one shared time range without adding events."""
    source = sorted(
        events, key=lambda item: (item.start_sample, item.channel, item.end_sample)
    )
    linked: list[Phase1Event] = []
    consumed: set[int] = set()
    for index, event in enumerate(source):
        if index in consumed:
            continue
        candidates = [
            (other_index, other)
            for other_index, other in enumerate(source)
            if other_index not in consumed
            and other_index != index
            and other.step == event.step
            and other.channel != event.channel
            and other.start_sample <= event.end_sample + max_gap_samples
            and event.start_sample <= other.end_sample + max_gap_samples
        ]
        if not candidates:
            linked.append(event)
            consumed.add(index)
            continue

        other_index, other = min(
            candidates,
            key=lambda item: abs(
                (item[1].start_sample + item[1].end_sample)
                - (event.start_sample + event.end_sample)
            ),
        )
        shared_start = min(event.start_sample, other.start_sample)
        shared_end = max(event.end_sample, other.end_sample)
        if (
            max_linked_duration_samples is not None
            and shared_end - shared_start > max_linked_duration_samples
        ):
            linked.append(event)
            consumed.add(index)
            continue

        attenuation_limit = None
        if (
            max_attenuation_delta_db is not None
            and event.action == "repaired"
            and other.action == "repaired"
        ):
            attenuation_limit = (
                min(event.attenuation_db, other.attenuation_db)
                + max_attenuation_delta_db
            )

        for original in (event, other):
            linked.append(
                Phase1Event(
                    step=original.step,
                    channel=original.channel,
                    start_sample=shared_start,
                    end_sample=shared_end,
                    confidence=original.confidence,
                    action=original.action,
                    score=original.score,
                    attenuation_db=(
                        min(original.attenuation_db, attenuation_limit)
                        if attenuation_limit is not None
                        else original.attenuation_db
                    ),
                )
            )
        consumed.update((index, other_index))
    return sorted(linked, key=lambda item: (item.channel, item.start_sample))


def _detect_plosives_in_chunk(
    samples: np.ndarray,
    sample_rate: int,
    config: Phase1Config,
    channel: int,
    offset: int,
) -> list[Phase1Event]:
    nyquist = sample_rate / 2
    if config.plosive_mid_high_hz >= nyquist:
        return []

    low_sos = butter(
        3, config.plosive_low_hz, btype="lowpass", fs=sample_rate, output="sos"
    )
    mid_sos = butter(
        3,
        (config.plosive_mid_low_hz, config.plosive_mid_high_hz),
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    low = _safe_sosfiltfilt(low_sos, samples)
    mid = _safe_sosfiltfilt(mid_sos, samples)

    frame_size = max(8, round(config.plosive_frame_ms * sample_rate / 1000))
    hop_size = max(1, round(config.plosive_hop_ms * sample_rate / 1000))
    low_rms, centers = _frame_rms(low, frame_size, hop_size)
    mid_rms, _ = _frame_rms(mid, frame_size, hop_size)
    epsilon = 1e-9
    low_db = 20 * np.log10(low_rms + epsilon)
    mid_db = 20 * np.log10(mid_rms + epsilon)
    ratio_db = low_db - mid_db

    baseline_frames = max(
        3, round(config.plosive_baseline_sec * sample_rate / hop_size)
    )
    if baseline_frames % 2 == 0:
        baseline_frames += 1
    baseline = median_filter(low_db, size=baseline_frames, mode="nearest")
    excess_db = low_db - baseline

    candidate = (
        (excess_db >= config.plosive_candidate_excess_db)
        & (ratio_db >= config.plosive_candidate_ratio_db)
        & (low_db >= config.plosive_min_level_dbfs)
    )
    spans = _mask_spans(candidate, max_gap=max(1, round(15 / config.plosive_hop_ms)))
    events: list[Phase1Event] = []
    lead = round(10 * sample_rate / 1000)
    tail = round(35 * sample_rate / 1000)

    for first_frame, last_frame in spans:
        selection = slice(first_frame, last_frame)
        peak_excess = float(np.max(excess_db[selection]))
        peak_ratio = float(np.max(ratio_db[selection]))
        peak_level = float(np.max(low_db[selection]))
        raw_start = int(centers[first_frame] - frame_size // 2)
        raw_end = int(centers[last_frame - 1] + frame_size // 2)
        start = max(0, raw_start - lead)
        end = min(samples.size, raw_end + tail)
        duration_ms = (end - start) * 1000 / sample_rate

        repair = (
            peak_excess >= config.plosive_repair_excess_db
            and peak_ratio >= config.plosive_repair_ratio_db
            and duration_ms <= config.plosive_max_duration_ms
        )
        confidence = float(
            np.clip(
                0.35
                + 0.035 * (peak_excess - config.plosive_candidate_excess_db)
                + 0.02 * (peak_ratio - config.plosive_candidate_ratio_db),
                0.35,
                0.99,
            )
        )
        attenuation_db = (
            min(
                config.plosive_max_attenuation_db,
                3.0 + max(0.0, peak_excess - config.plosive_repair_excess_db) * 0.45,
            )
            if repair
            else 0.0
        )
        score = peak_excess + max(0.0, peak_ratio)
        events.append(
            Phase1Event(
                step="de_plosive",
                channel=channel,
                start_sample=offset + start,
                end_sample=offset + end,
                confidence=confidence,
                action="repaired" if repair else "review",
                score=score,
                attenuation_db=attenuation_db,
            )
        )
    return events


def detect_plosives(
    audio: np.ndarray, sample_rate: int, config: Phase1Config
) -> list[Phase1Event]:
    chunk_size = max(1, round(config.analysis_chunk_sec * sample_rate))
    padding = round(0.4 * sample_rate)
    events: list[Phase1Event] = []
    total_samples = audio.shape[1]

    for channel in range(2):
        for core_start in range(0, total_samples, chunk_size):
            core_end = min(total_samples, core_start + chunk_size)
            analysis_start = max(0, core_start - padding)
            analysis_end = min(total_samples, core_end + padding)
            chunk_events = _detect_plosives_in_chunk(
                audio[channel, analysis_start:analysis_end],
                sample_rate,
                config,
                channel,
                analysis_start,
            )
            events.extend(
                event
                for event in chunk_events
                if core_start
                <= (event.start_sample + event.end_sample) // 2
                < core_end
            )
    deduplicated = _deduplicate_events(events)
    return _link_stereo_events(
        deduplicated,
        max_gap_samples=round(20 * sample_rate / 1000),
        max_linked_duration_samples=round(
            config.plosive_max_duration_ms * sample_rate / 1000
        ),
        max_attenuation_delta_db=config.plosive_max_lr_delta_db,
    )


def repair_plosives(
    audio: np.ndarray,
    sample_rate: int,
    events: Iterable[Phase1Event],
    config: Phase1Config,
    *,
    copy: bool = True,
) -> np.ndarray:
    repaired = audio.copy() if copy else audio
    low_sos = butter(
        3, config.plosive_low_hz, btype="lowpass", fs=sample_rate, output="sos"
    )
    filter_padding = round(60 * sample_rate / 1000)
    fade_samples = max(1, round(8 * sample_rate / 1000))

    for event in events:
        if event.step != "de_plosive" or event.action != "repaired":
            continue
        local_start = max(0, event.start_sample - filter_padding)
        local_end = min(audio.shape[1], event.end_sample + filter_padding)
        segment = repaired[event.channel, local_start:local_end].copy()
        low = _safe_sosfiltfilt(low_sos, segment)

        event_start = event.start_sample - local_start
        event_end = event.end_sample - local_start
        envelope = np.zeros(segment.size, dtype=np.float32)
        event_length = max(0, event_end - event_start)
        event_fade = min(fade_samples, event_length // 2)
        if event_length > 0:
            envelope[event_start:event_end] = 1.0
        if event_fade > 0:
            envelope[event_start : event_start + event_fade] = np.linspace(
                0.0, 1.0, event_fade, endpoint=False
            )
            envelope[event_end - event_fade : event_end] = np.linspace(
                1.0, 0.0, event_fade, endpoint=False
            )

        low_gain = 10 ** (-event.attenuation_db / 20)
        repaired[event.channel, local_start:local_end] = (
            segment + (low_gain - 1.0) * low * envelope
        )
    return repaired


def _detect_clicks_in_chunk(
    samples: np.ndarray,
    sample_rate: int,
    config: Phase1Config,
    channel: int,
    offset: int,
) -> list[Phase1Event]:
    if config.click_highpass_hz >= sample_rate / 2:
        return []

    high_sos = butter(
        3, config.click_highpass_hz, btype="highpass", fs=sample_rate, output="sos"
    )
    high = _safe_sosfiltfilt(high_sos, samples)
    absolute = np.abs(high)
    baseline_size = max(3, round(config.click_baseline_ms * sample_rate / 1000))
    local_baseline = uniform_filter1d(
        absolute, size=baseline_size, mode="nearest"
    )
    global_baseline = float(np.median(absolute))
    absolute_floor = 10 ** (config.click_min_level_dbfs / 20)
    threshold = np.maximum(
        local_baseline * config.click_candidate_ratio,
        max(global_baseline * config.click_candidate_ratio, absolute_floor),
    )

    curvature = np.zeros_like(samples)
    if samples.size > 2:
        curvature[1:-1] = np.abs(
            samples[2:] - 2.0 * samples[1:-1] + samples[:-2]
        )
    candidate = (absolute > threshold) & (
        curvature > np.maximum(local_baseline * 3.5, absolute_floor)
    )
    join_gap = max(1, round(0.35 * sample_rate / 1000))
    spans = _mask_spans(candidate, max_gap=join_gap)
    events: list[Phase1Event] = []
    repair_margin = max(1, round(0.08 * sample_rate / 1000))

    for raw_start, raw_end in spans:
        duration_ms = (raw_end - raw_start) * 1000 / sample_rate
        if duration_ms > config.click_max_review_ms:
            continue
        region = slice(raw_start, raw_end)
        ratio = float(
            np.max(absolute[region] / np.maximum(threshold[region], 1e-12))
            * config.click_candidate_ratio
        )
        start = max(0, raw_start - repair_margin)
        end = min(samples.size, raw_end + repair_margin)
        repair = (
            ratio >= config.click_repair_ratio
            and (end - start) * 1000 / sample_rate <= config.click_max_repair_ms
            and start >= 2
            and end + 2 < samples.size
        )
        confidence = float(
            np.clip(
                0.35
                + 0.65
                * (ratio - config.click_candidate_ratio)
                / max(1.0, config.click_repair_ratio),
                0.35,
                0.99,
            )
        )
        events.append(
            Phase1Event(
                step="mouth_de_click",
                channel=channel,
                start_sample=offset + start,
                end_sample=offset + end,
                confidence=confidence,
                action="repaired" if repair else "review",
                score=ratio,
            )
        )
    return events


def detect_clicks(
    audio: np.ndarray, sample_rate: int, config: Phase1Config
) -> list[Phase1Event]:
    chunk_size = max(1, round(config.analysis_chunk_sec * sample_rate))
    padding = round(0.05 * sample_rate)
    events: list[Phase1Event] = []
    total_samples = audio.shape[1]

    for channel in range(2):
        for core_start in range(0, total_samples, chunk_size):
            core_end = min(total_samples, core_start + chunk_size)
            analysis_start = max(0, core_start - padding)
            analysis_end = min(total_samples, core_end + padding)
            chunk_events = _detect_clicks_in_chunk(
                audio[channel, analysis_start:analysis_end],
                sample_rate,
                config,
                channel,
                analysis_start,
            )
            events.extend(
                event
                for event in chunk_events
                if core_start
                <= (event.start_sample + event.end_sample) // 2
                < core_end
            )
    deduplicated = _deduplicate_events(events)
    return _link_stereo_events(
        deduplicated,
        max_gap_samples=round(0.5 * sample_rate / 1000),
        max_linked_duration_samples=round(
            config.click_max_repair_ms * sample_rate / 1000
        ),
    )


def repair_clicks(
    audio: np.ndarray,
    events: Iterable[Phase1Event],
    *,
    copy: bool = True,
) -> np.ndarray:
    repaired = audio.copy() if copy else audio
    for event in events:
        if event.step != "mouth_de_click" or event.action != "repaired":
            continue
        start = event.start_sample
        end = event.end_sample
        channel = event.channel
        if start < 2 or end + 2 >= repaired.shape[1] or end <= start:
            continue

        samples = repaired[channel]
        left_value = float(samples[start - 1])
        right_value = float(samples[end])
        left_slope = float((samples[start - 1] - samples[start - 2]) * 0.5)
        right_slope = float((samples[end + 1] - samples[end]) * 0.5)
        length = end - start + 1
        t = np.arange(1, length, dtype=np.float64) / length
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        interpolated = (
            h00 * left_value
            + h10 * length * left_slope
            + h01 * right_value
            + h11 * length * right_slope
        )
        context = samples[max(0, start - 8) : min(samples.size, end + 8)]
        limit = max(1e-6, float(np.max(np.abs(context))) * 1.1)
        samples[start:end] = np.clip(interpolated, -limit, limit)
    return repaired


def _audio_metrics(audio: np.ndarray) -> dict[str, Any]:
    def channel_rms(channel: np.ndarray) -> float:
        square_sum = 0.0
        block_size = 1024 * 1024
        for start in range(0, channel.size, block_size):
            block = channel[start : start + block_size]
            square_sum += float(np.sum(block * block, dtype=np.float64))
        return math.sqrt(square_sum / channel.size)

    return {
        "peak": [round(float(np.max(np.abs(channel))), 8) for channel in audio],
        "rms": [
            round(channel_rms(channel), 8)
            for channel in audio
        ],
    }


def process_phase1(
    audio: np.ndarray,
    sample_rate: int,
    config: Phase1Config | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or Phase1Config.conservative()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 2:
        channels = audio.shape[0] if audio.ndim == 2 else "unknown"
        raise Phase1Error(
            f"Phase 1はバイノーラル2ch専用です（入力: {channels}ch）。"
        )
    if sample_rate < 16000:
        raise Phase1Error(
            f"Phase 1は16kHz以上の音声が必要です（入力: {sample_rate}Hz）。"
        )
    if audio.shape[1] < max(64, round(sample_rate * 0.05)):
        raise Phase1Error("Phase 1には50ms以上の音声が必要です。")
    if not np.all(np.isfinite(audio)):
        raise Phase1Error("入力音声にNaNまたはInfが含まれています。")

    processed = audio.copy()
    plosive_events = detect_plosives(processed, sample_rate, config)
    repair_plosives(
        processed, sample_rate, plosive_events, config, copy=False
    )
    click_events = detect_clicks(processed, sample_rate, config)
    repair_clicks(processed, click_events, copy=False)
    if processed.shape != audio.shape:
        raise RuntimeError("Phase 1処理でチャンネル数またはサンプル数が変化しました。")
    if not np.all(np.isfinite(processed)):
        raise RuntimeError("Phase 1処理結果にNaNまたはInfが含まれています。")

    events = sorted(
        plosive_events + click_events,
        key=lambda event: (event.start_sample, event.channel, event.step),
    )
    report = {
        "phase": "phase1",
        "version": PHASE1_VERSION,
        "steps": list(PHASE1_STEPS),
        "sample_rate": sample_rate,
        "channels": 2,
        "samples": int(audio.shape[1]),
        "config": asdict(config),
        "metrics": {
            "before": _audio_metrics(audio),
            "after": _audio_metrics(processed),
        },
        "summary": {
            "de_plosive": sum(
                event.action == "repaired"
                for event in plosive_events
            ),
            "mouth_de_click": sum(
                event.action == "repaired"
                for event in click_events
            ),
            "review": sum(event.action == "review" for event in events),
        },
        "events": [event.to_dict(sample_rate) for event in events],
        "warnings": [],
    }
    if np.max(np.abs(processed)) > 1.0:
        report["warnings"].append(
            "処理結果のピークが0 dBFSを超えています。Phase 1では自動ノーマライズしません。"
        )
    return processed.astype(np.float32, copy=False), report


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_audio_atomic(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".wav", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        sf.write(
            str(temporary_path),
            audio.T,
            sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def process_phase1_file(
    input_path: Path,
    output_path: Path,
    config: Phase1Config | None = None,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    config = config or Phase1Config.conservative()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise Phase1Error("Phase 1は元ファイルへ上書きできません。")
    if output_path.suffix.lower() != ".wav":
        raise Phase1Error("Phase 1の出力形式はWAVを指定してください。")

    input_stat = input_path.stat()
    input_hash = file_sha256(input_path)
    info = sf.info(str(input_path))
    if info.channels != 2:
        raise Phase1Error(
            f"Phase 1はバイノーラル2ch専用です（{input_path.name}: {info.channels}ch）。"
        )
    interleaved, sample_rate = sf.read(
        str(input_path), dtype="float32", always_2d=True
    )
    processed, report = process_phase1(interleaved.T, sample_rate, config)
    final_input_stat = input_path.stat()
    if (
        final_input_stat.st_size != input_stat.st_size
        or final_input_stat.st_mtime_ns != input_stat.st_mtime_ns
    ):
        raise Phase1Error(
            f"処理中に入力ファイルが変更されました。再実行してください: {input_path.name}"
        )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path.resolve()),
                "size": input_stat.st_size,
                "mtime_ns": input_stat.st_mtime_ns,
                "sha256": input_hash,
                "format": info.format,
                "subtype": info.subtype,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "samples": info.frames,
            },
            "output": {
                "path": str(output_path.resolve()),
                "format": "WAV",
                "subtype": "FLOAT",
                "sample_rate": sample_rate,
                "channels": 2,
                "samples": int(processed.shape[1]),
            },
        }
    )
    _write_audio_atomic(output_path, processed, sample_rate)
    output_stat = output_path.stat()
    report["output"].update(
        {
            "size": output_stat.st_size,
            "sha256": file_sha256(output_path),
        }
    )
    if write_report:
        _write_json_atomic(sidecar_path_for(output_path), report)
    return report


def phase1_is_current(
    input_path: Path,
    output_path: Path,
    config: Phase1Config | None = None,
) -> bool:
    config = config or Phase1Config.conservative()
    report_path = sidecar_path_for(output_path)
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        input_data = report["input"]
        stat = input_path.stat()
        if (
            report.get("phase") != "phase1"
            or report.get("version") != PHASE1_VERSION
            or report.get("steps") != list(PHASE1_STEPS)
            or report.get("config") != asdict(config)
            or input_data.get("size") != stat.st_size
            or input_data.get("mtime_ns") != stat.st_mtime_ns
        ):
            return False
        output_data = report["output"]
        output_stat = output_path.stat()
        if output_data.get("size") != output_stat.st_size:
            return False
        return (
            input_data.get("sha256") == file_sha256(input_path)
            and output_data.get("sha256") == file_sha256(output_path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
