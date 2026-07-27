import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import app
import phase1
from phase1 import (
    Phase1Config,
    Phase1Error,
    Phase1Event,
    _link_stereo_events,
    phase1_is_current,
    process_phase1,
    process_phase1_file,
    repair_clicks,
    repair_plosives,
    sidecar_path_for,
)


SAMPLE_RATE = 48000


def synthetic_binaural() -> tuple[np.ndarray, int, int, tuple[int, int]]:
    sample_count = SAMPLE_RATE * 2
    time = np.arange(sample_count) / SAMPLE_RATE
    voice = (
        0.025 * np.sin(2 * np.pi * 220 * time)
        + 0.008 * np.sin(2 * np.pi * 330 * time)
    )
    audio = np.vstack([voice.copy(), 0.7 * voice.copy()]).astype(np.float32)

    plosive_start = round(0.75 * SAMPLE_RATE)
    plosive_end = round(0.86 * SAMPLE_RATE)
    window = np.hanning(plosive_end - plosive_start)
    burst_time = np.arange(plosive_end - plosive_start) / SAMPLE_RATE
    audio[0, plosive_start:plosive_end] += (
        0.65 * np.sin(2 * np.pi * 75 * burst_time) * window
    )

    click_sample = round(1.35 * SAMPLE_RATE)
    audio[1, click_sample] = 0.8
    audio[1, click_sample + 1] = -0.55
    return audio, plosive_start, click_sample, (plosive_start, plosive_end)


class Phase1ProcessingTests(unittest.TestCase):
    def test_phase1_runs_steps_in_fixed_order(self):
        audio = np.zeros((2, SAMPLE_RATE // 2), dtype=np.float32)
        calls = []

        def detect_plosives(*args):
            calls.append("detect_plosives")
            return []

        def repair_plosives(*args, **kwargs):
            calls.append("repair_plosives")
            return args[0]

        def detect_clicks(*args):
            calls.append("detect_clicks")
            return []

        def repair_clicks(*args, **kwargs):
            calls.append("repair_clicks")
            return args[0]

        with (
            patch("phase1.detect_plosives", side_effect=detect_plosives),
            patch("phase1.repair_plosives", side_effect=repair_plosives),
            patch("phase1.detect_clicks", side_effect=detect_clicks),
            patch("phase1.repair_clicks", side_effect=repair_clicks),
        ):
            _, report = process_phase1(audio, SAMPLE_RATE)

        self.assertEqual(
            calls,
            [
                "detect_plosives",
                "repair_plosives",
                "detect_clicks",
                "repair_clicks",
            ],
        )
        self.assertEqual(report["steps"], ["de_plosive", "mouth_de_click"])

    def test_synthetic_plosive_and_click_are_repaired(self):
        audio, _, click_sample, plosive_range = synthetic_binaural()
        processed, report = process_phase1(audio, SAMPLE_RATE)

        self.assertEqual(processed.shape, audio.shape)
        self.assertEqual(report["summary"]["de_plosive"], 1)
        self.assertEqual(report["summary"]["mouth_de_click"], 1)
        self.assertLess(
            np.max(np.abs(processed[1, click_sample - 2 : click_sample + 4])),
            np.max(np.abs(audio[1, click_sample - 2 : click_sample + 4])) * 0.1,
        )
        start, end = plosive_range
        before = np.sqrt(np.mean(audio[0, start:end] ** 2))
        after = np.sqrt(np.mean(processed[0, start:end] ** 2))
        self.assertLess(after, before)
        self.assertTrue(
            np.array_equal(
                audio[:, : SAMPLE_RATE // 2],
                processed[:, : SAMPLE_RATE // 2],
            )
        )
        for channel_index, channel_name in enumerate(("L", "R")):
            repaired_mask = np.zeros(audio.shape[1], dtype=bool)
            for event in report["events"]:
                if event["channel"] == channel_name and event["action"] == "repaired":
                    repaired_mask[
                        event["start_sample"] : event["end_sample"]
                    ] = True
            self.assertTrue(
                np.array_equal(
                    audio[channel_index, ~repaired_mask],
                    processed[channel_index, ~repaired_mask],
                )
            )

    def test_plosive_repair_does_not_touch_other_channel(self):
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.01, (2, SAMPLE_RATE)).astype(np.float32)
        event = Phase1Event(
            step="de_plosive",
            channel=0,
            start_sample=12000,
            end_sample=16000,
            confidence=0.9,
            action="repaired",
            score=20.0,
            attenuation_db=8.0,
        )

        repaired = repair_plosives(
            audio, SAMPLE_RATE, [event], Phase1Config.conservative()
        )

        self.assertTrue(np.array_equal(repaired[1], audio[1]))
        self.assertTrue(np.array_equal(repaired[0, :8000], audio[0, :8000]))
        self.assertTrue(np.array_equal(repaired[0, 20000:], audio[0, 20000:]))

    def test_click_repair_keeps_sample_count_and_other_channel(self):
        audio = np.zeros((2, 1000), dtype=np.float32)
        audio[0, 500:502] = (0.9, -0.7)
        event = Phase1Event(
            step="mouth_de_click",
            channel=0,
            start_sample=500,
            end_sample=502,
            confidence=0.95,
            action="repaired",
            score=20.0,
        )

        repaired = repair_clicks(audio, [event])

        self.assertEqual(repaired.shape, audio.shape)
        self.assertTrue(np.array_equal(repaired[1], audio[1]))
        self.assertLess(np.max(np.abs(repaired[0, 500:502])), 0.01)

    def test_matching_left_and_right_events_share_boundaries(self):
        events = [
            Phase1Event(
                step="de_plosive",
                channel=0,
                start_sample=100,
                end_sample=200,
                confidence=0.9,
                action="repaired",
                score=20.0,
                attenuation_db=9.0,
            ),
            Phase1Event(
                step="de_plosive",
                channel=1,
                start_sample=110,
                end_sample=210,
                confidence=0.8,
                action="repaired",
                score=18.0,
                attenuation_db=3.0,
            ),
        ]

        linked = _link_stereo_events(
            events,
            max_gap_samples=20,
            max_attenuation_delta_db=3.0,
        )

        self.assertEqual(
            [(event.start_sample, event.end_sample) for event in linked],
            [(100, 210), (100, 210)],
        )
        self.assertEqual(
            [event.attenuation_db for event in linked],
            [6.0, 3.0],
        )

    def test_mono_input_is_rejected(self):
        with self.assertRaisesRegex(Phase1Error, "2ch専用"):
            process_phase1(
                np.zeros((1, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE
            )


class Phase1FileTests(unittest.TestCase):
    def test_common_binaural_master_formats_preserve_rate_and_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sample_rate, subtype in product(
                (48000, 96000), ("PCM_16", "PCM_24", "FLOAT")
            ):
                with self.subTest(sample_rate=sample_rate, subtype=subtype):
                    source = root / f"source-{sample_rate}-{subtype}.wav"
                    output = root / f"output-{sample_rate}-{subtype}.wav"
                    frames = sample_rate // 10
                    sf.write(
                        source,
                        np.zeros((frames, 2), dtype=np.float32),
                        sample_rate,
                        subtype=subtype,
                    )

                    process_phase1_file(source, output)
                    output_info = sf.info(output)

                    self.assertEqual(output_info.channels, 2)
                    self.assertEqual(output_info.samplerate, sample_rate)
                    self.assertEqual(output_info.frames, frames)
                    self.assertEqual(output_info.subtype, "FLOAT")

    def test_file_output_is_float_wav_with_sidecar_and_resume_identity(self):
        audio = np.zeros((SAMPLE_RATE // 5, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            output = root / "phase1" / "source.wav"
            sf.write(source, audio, SAMPLE_RATE, subtype="PCM_24")

            report = process_phase1_file(source, output)
            output_info = sf.info(output)
            sidecar = sidecar_path_for(output)
            saved_report = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertEqual(output_info.channels, 2)
            self.assertEqual(output_info.samplerate, SAMPLE_RATE)
            self.assertEqual(output_info.frames, audio.shape[0])
            self.assertEqual(output_info.subtype, "FLOAT")
            self.assertEqual(report["phase"], "phase1")
            self.assertEqual(saved_report["steps"], ["de_plosive", "mouth_de_click"])
            self.assertTrue(phase1_is_current(source, output))

            with output.open("r+b") as handle:
                handle.seek(-4, 2)
                handle.write(b"\x7f\x7f\x7f\x7f")
            self.assertFalse(phase1_is_current(source, output))

            process_phase1_file(source, output)
            self.assertTrue(phase1_is_current(source, output))

            sf.write(source, audio + 0.001, SAMPLE_RATE, subtype="PCM_24")
            self.assertFalse(phase1_is_current(source, output))

    def test_file_rejects_non_stereo_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "mono.wav"
            output = root / "out.wav"
            sf.write(
                source,
                np.zeros(SAMPLE_RATE // 5, dtype=np.float32),
                SAMPLE_RATE,
            )

            with self.assertRaisesRegex(Phase1Error, "2ch専用"):
                process_phase1_file(source, output)

            self.assertFalse(output.exists())

    def test_phase1_output_path_is_always_wav(self):
        source_root = Path("/input")
        output_root = Path("/output")
        source = source_root / "scene" / "take.flac"

        result = app.phase1_output_path_for(source, source_root, output_root)

        self.assertEqual(result, output_root / "scene" / "take.wav")

    def test_output_name_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "take.wav"
            second = second_dir / "take.flac"
            sf.write(first, np.zeros((100, 2)), SAMPLE_RATE)
            sf.write(second, np.zeros((100, 2)), SAMPLE_RATE)

            with self.assertRaisesRegex(app.gr.Error, "同じ出力名"):
                app.resolve_jobs(
                    "drop",
                    [str(first), str(second)],
                    "",
                    str(root / "output"),
                    phase1=True,
                )

    def test_batch_continues_after_incompatible_file_and_resumes_valid_file(self):
        class ImmediateProgress:
            @staticmethod
            def tqdm(jobs, desc):
                return jobs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "raw"
            output_dir = root / "phase1"
            input_dir.mkdir()
            sf.write(
                input_dir / "stereo.wav",
                np.zeros((SAMPLE_RATE // 5, 2), dtype=np.float32),
                SAMPLE_RATE,
            )
            sf.write(
                input_dir / "mono.wav",
                np.zeros(SAMPLE_RATE // 5, dtype=np.float32),
                SAMPLE_RATE,
            )

            first = app.run_phase1_batch(
                "folder",
                None,
                str(input_dir),
                str(output_dir),
                progress=ImmediateProgress(),
            )
            second = app.run_phase1_batch(
                "folder",
                None,
                str(input_dir),
                str(output_dir),
                progress=ImmediateProgress(),
            )

            self.assertIn("Phase 1完了: 1件", first)
            self.assertIn("失敗: 1件", first)
            self.assertIn("スキップ(同じ入力・設定): 1件", second)
            self.assertTrue((output_dir / "stereo.wav").is_file())
            self.assertTrue((output_dir / "stereo.phase1.json").is_file())


if __name__ == "__main__":
    unittest.main()
