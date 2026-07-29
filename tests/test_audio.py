"""Step 1 tests: the waveform invariants everything downstream assumes."""

from __future__ import annotations

import math

import pytest
import torch

from whispr import audio


def dominant_freq(wav: torch.Tensor, sr: int) -> float:
    """Peak of the magnitude spectrum — our 'what note is this' oracle."""
    spec = torch.fft.rfft(wav * torch.hann_window(len(wav)))
    return float(torch.argmax(spec.abs())) * sr / len(wav)


class TestSynthesis:
    @pytest.mark.parametrize("freq", [100.0, 440.0, 1000.0, 4000.0])
    def test_sine_has_the_frequency_it_claims(self, freq):
        sr = 16_000
        wav = audio.sine(freq, 1.0, sr)
        assert dominant_freq(wav, sr) == pytest.approx(freq, abs=2.0)

    def test_chirp_sweeps_at_the_right_rate(self):
        """The classic bug is sin(2*pi*f(t)*t), which sweeps twice too fast."""
        sr = 16_000
        c = audio.chirp(500, 3000, 2.0, sr)
        # First and last 0.2 s should sit near the endpoints of the sweep.
        assert dominant_freq(c[: sr // 5], sr) == pytest.approx(625, abs=80)
        assert dominant_freq(c[-sr // 5 :], sr) == pytest.approx(2875, abs=80)

    def test_amplitude_is_respected(self):
        assert audio.sine(440, 0.1, amplitude=0.3).abs().max() == pytest.approx(0.3, abs=1e-3)


class TestAliasing:
    @pytest.mark.parametrize(
        "freq,sr,expected",
        [
            (1000, 16_000, 1000),  # below Nyquist: unchanged
            (7999, 16_000, 7999),  # just below Nyquist: unchanged
            (9000, 16_000, 7000),  # folds
            (15_000, 16_000, 1000),  # folds near DC
            (17_000, 16_000, 1000),  # past sr, wraps
            (3000, 4_000, 1000),  # the figure in notes/01
        ],
    )
    def test_foldback_closed_form(self, freq, sr, expected):
        assert audio.alias_frequency(freq, sr) == pytest.approx(expected, abs=1e-6)

    def test_aliasing_actually_happens_when_you_slice(self):
        """A 3 kHz tone naively decimated 16k->4k really does become 1 kHz."""
        sr = 16_000
        wav = audio.sine(3000, 1.0, sr)
        naive = wav[::4]  # no anti-alias filter
        assert dominant_freq(naive, 4_000) == pytest.approx(1000, abs=15)

    def test_proper_resampling_suppresses_the_alias(self):
        """torchaudio low-pass filters first, so the 3 kHz tone is removed, not folded.

        We measure the steady-state interior: the filter has a boundary
        transient at each end (the first output sample is ~0.12) which is a
        property of the filter's edge handling, not of aliasing.
        """
        import torchaudio

        sr = 16_000
        wav = audio.sine(3000, 1.0, sr)
        proper = torchaudio.transforms.Resample(sr, 4_000)(wav)

        interior = proper[200:-200]
        assert interior.abs().max() < 0.05 * wav.abs().max()

    def test_proper_resampling_leaves_no_alias_peak(self):
        """The spectral version of the claim: no energy shows up at 1 kHz."""
        import torchaudio

        sr = 16_000
        wav = audio.sine(3000, 1.0, sr)
        naive = wav[::4]
        proper = torchaudio.transforms.Resample(sr, 4_000)(wav)[200:-200]

        def energy_at(sig, freq, out_sr=4_000):
            spec = (torch.fft.rfft(sig * torch.hann_window(len(sig)))).abs()
            bin_ = round(freq * len(sig) / out_sr)
            return spec[bin_ - 2 : bin_ + 3].max().item() / len(sig)

        # Naive decimation parks a big peak at 1 kHz; proper resampling doesn't.
        assert energy_at(naive, 1000) > 0.1
        assert energy_at(proper, 1000) < 0.005


class TestPadOrTrim:
    def test_pads_short_audio_with_zeros(self):
        wav = audio.sine(440, 1.0)
        out = audio.pad_or_trim(wav)
        assert out.shape == (audio.N_SAMPLES,)
        assert torch.all(out[audio.SAMPLE_RATE :] == 0)

    def test_trims_long_audio(self):
        wav = audio.sine(440, 40.0)
        assert audio.pad_or_trim(wav).shape == (audio.N_SAMPLES,)

    def test_exact_length_is_untouched(self):
        wav = torch.randn(audio.N_SAMPLES)
        assert torch.equal(audio.pad_or_trim(wav), wav)

    def test_thirty_seconds_is_480k(self):
        """The constant the encoder's 1500 positions depend on."""
        assert audio.N_SAMPLES == 480_000


class TestIO:
    def test_roundtrip_preserves_signal(self, tmp_path):
        wav = audio.sine(440, 0.5, amplitude=0.8)
        path = tmp_path / "t.wav"
        audio.save_audio(path, wav)
        back = audio.load_audio(path)
        assert back.shape == wav.shape
        assert back.dtype == torch.float32
        # 16-bit PCM quantization is the only loss.
        assert (back - wav).abs().max() < 2 / 32768

    def test_load_resamples_to_16k(self, tmp_path):
        path = tmp_path / "t.wav"
        audio.save_audio(path, audio.sine(440, 0.5, sample_rate=44_100), 44_100)
        back = audio.load_audio(path)
        assert back.shape[0] == pytest.approx(0.5 * 16_000, rel=0.01)
        assert dominant_freq(back, 16_000) == pytest.approx(440, abs=5)

    def test_stereo_is_downmixed_to_mono(self, tmp_path):
        path = tmp_path / "t.wav"
        stereo = torch.stack([audio.sine(440, 0.2), audio.sine(880, 0.2)])
        audio.save_audio(path, stereo)
        assert audio.load_audio(path).dim() == 1


class TestQuantize:
    def test_more_bits_means_less_error(self):
        wav = audio.sine(440, 0.05, amplitude=0.9)
        errors = [(audio.quantize(wav, b) - wav).abs().mean().item() for b in (3, 6, 10, 16)]
        assert errors == sorted(errors, reverse=True)

    def test_sixteen_bit_error_is_negligible(self):
        wav = audio.sine(440, 0.05, amplitude=0.9)
        assert (audio.quantize(wav, 16) - wav).abs().max() < 1e-4


def test_db_is_twenty_log10():
    assert audio.db(torch.tensor([1.0])).item() == pytest.approx(0.0)
    assert audio.db(torch.tensor([0.1])).item() == pytest.approx(-20.0)
    assert audio.db(torch.tensor([0.0])).item() == pytest.approx(20 * math.log10(1e-10))
