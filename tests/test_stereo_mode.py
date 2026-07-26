import tempfile
import unittest
from unittest.mock import call, patch
from pathlib import Path

import soundfile as sf
import torch

import app


class StereoModeTests(unittest.TestCase):
    def test_open_in_file_manager_uses_open_on_macos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("app.sys.platform", "darwin"), patch("app.subprocess.Popen") as launch:
                status = app.open_in_file_manager(temp_dir)

        launch.assert_called_once_with(["open", str(Path(temp_dir).resolve())])
        self.assertIn("フォルダを開きました", status)

    def test_denoise_file_writes_two_channels_in_preserving_mode(self):
        audio = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            sf.write(str(input_path), audio.T.numpy(), 44100)

            with patch("app._denoise_resemble_channel", side_effect=lambda channel, *_: channel):
                app.denoise_file(
                    input_path,
                    output_path,
                    strength_db=100,
                    engine="re_denoise",
                    preserve_stereo=True,
                )

            written, _ = app.torchaudio.load(str(output_path))

        self.assertEqual(tuple(written.shape), tuple(audio.shape))

    def test_preserving_mode_processes_each_channel_without_downmixing(self):
        audio = torch.tensor([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ])

        def fake_channel_denoise(channel, sr, do_enhance):
            return channel + 100

        with patch("app._denoise_resemble_channel", side_effect=fake_channel_denoise) as denoise:
            cleaned = app._denoise_resemble(
                audio, sr=44100, do_enhance=False, preserve_stereo=True
            )

        self.assertTrue(torch.equal(cleaned, audio + 100))
        self.assertEqual(len(denoise.call_args_list), 2)
        for actual, expected in zip(
            denoise.call_args_list,
            [call(audio[0], 44100, False), call(audio[1], 44100, False)],
        ):
            self.assertTrue(torch.equal(actual.args[0], expected.args[0]))
            self.assertEqual(actual.args[1:], expected.args[1:])

    def test_default_mode_keeps_existing_mono_downmix(self):
        audio = torch.tensor([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ])

        with patch("app._denoise_resemble_channel", side_effect=lambda channel, *_: channel) as denoise:
            cleaned = app._denoise_resemble(
                audio, sr=44100, do_enhance=False, preserve_stereo=False
            )

        self.assertTrue(torch.equal(cleaned, audio.mean(0, keepdim=True)))
        self.assertEqual(denoise.call_count, 1)
        actual = denoise.call_args
        self.assertTrue(torch.equal(actual.args[0], audio.mean(0)))
        self.assertEqual(actual.args[1:], (44100, False))

    def test_mono_input_is_unchanged_by_preserving_mode(self):
        audio = torch.tensor([[1.0, 2.0, 3.0]])

        with patch("app._denoise_resemble_channel", side_effect=lambda channel, *_: channel) as denoise:
            cleaned = app._denoise_resemble(
                audio, sr=44100, do_enhance=True, preserve_stereo=True
            )

        self.assertTrue(torch.equal(cleaned, audio))
        self.assertEqual(denoise.call_count, 1)
        actual = denoise.call_args
        self.assertTrue(torch.equal(actual.args[0], audio.mean(0)))
        self.assertEqual(actual.args[1:], (44100, True))


if __name__ == "__main__":
    unittest.main()
