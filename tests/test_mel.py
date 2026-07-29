"""Step 3 tests: the frontend must be numerically identical to OpenAI's.

The filterbank tests use `assets/whisper_mel_filters.npz`, the actual array
shipped inside openai/whisper, which is committed to this repo. The
end-to-end tests use the `openai-whisper` package if it is installed and skip
otherwise:

    uv run --with openai-whisper --with "numba>=0.61" pytest tests/test_mel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from whispr import audio, mel

REFERENCE_NPZ = Path(__file__).resolve().parent.parent / "assets" / "whisper_mel_filters.npz"


@pytest.fixture(scope="module")
def reference_80():
    return mel.load_reference_filters(REFERENCE_NPZ, n_mels=80)


class TestMelScale:
    def test_zero_maps_to_zero(self):
        assert mel.hz_to_mel(0.0) == pytest.approx(0.0)
        assert mel.mel_to_hz(0.0) == pytest.approx(0.0)

    def test_breakpoint_is_1000hz_at_15mel(self):
        """The Slaney scale switches from linear to log exactly here."""
        assert mel.hz_to_mel(1000.0) == pytest.approx(15.0)
        assert mel.mel_to_hz(15.0) == pytest.approx(1000.0)

    def test_linear_below_the_breakpoint(self):
        """Below 1 kHz, mel is exactly f/(200/3)."""
        for hz in (100.0, 400.0, 999.0):
            assert mel.hz_to_mel(hz) == pytest.approx(hz / (200.0 / 3.0))

    @pytest.mark.parametrize("hz", [1.0, 50.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    def test_roundtrip(self, hz):
        assert mel.mel_to_hz(mel.hz_to_mel(hz)) == pytest.approx(hz, rel=1e-9)

    def test_is_monotonic(self):
        hz = np.linspace(0, 8000, 2000)
        assert np.all(np.diff(mel.hz_to_mel(hz)) > 0)

    def test_compresses_high_frequencies(self):
        """The whole point: equal Hz gaps are unequal mel gaps.

        A 100 Hz gap is 1.5 mel down at 100-200 Hz but only 0.21 mel up at
        7000-7100 Hz — a 7.3x compression.
        """
        low = mel.hz_to_mel(200.0) - mel.hz_to_mel(100.0)
        high = mel.hz_to_mel(7100.0) - mel.hz_to_mel(7000.0)
        assert low / high == pytest.approx(7.3, abs=0.2)

    def test_is_slaney_not_htk(self):
        """Guards against silently swapping in the HTK formula."""
        htk = 2595.0 * np.log10(1.0 + 4000.0 / 700.0)
        assert abs(mel.hz_to_mel(4000.0) - htk) > 1.0


class TestFilterbank:
    def test_matches_openai_whisper_exactly(self, reference_80):
        """The decisive test of this step."""
        ours = mel.mel_filterbank(n_mels=80)
        assert ours.shape == reference_80.shape == (80, 201)
        assert torch.allclose(ours, reference_80, atol=1e-6)
        assert (ours - reference_80).abs().max().item() < 1e-7

    def test_matches_openai_whisper_128(self):
        ours = mel.mel_filterbank(n_mels=128)
        ref = mel.load_reference_filters(REFERENCE_NPZ, n_mels=128)
        assert torch.allclose(ours, ref, atol=1e-6)

    def test_shape_is_nmels_by_bins(self):
        assert mel.mel_filterbank(n_mels=80, n_fft=400).shape == (80, 201)

    def test_all_weights_nonnegative(self):
        assert (mel.mel_filterbank() >= 0).all()

    def test_filters_are_triangular(self):
        """Each row rises to a single peak then falls — no second bump."""
        fb = mel.mel_filterbank()
        for i in range(fb.shape[0]):
            row = fb[i]
            nz = row.nonzero().flatten()
            if len(nz) < 3:
                continue
            seg = row[nz[0] : nz[-1] + 1]
            peak = int(torch.argmax(seg))
            assert torch.all(torch.diff(seg[: peak + 1]) >= -1e-6)
            assert torch.all(torch.diff(seg[peak:]) <= 1e-6)

    def test_adjacent_filters_overlap_except_where_undersampled(self):
        """Triangles half-overlap in *theory*, but at n_fft=400 the low bands
        are narrower than the 40 Hz bin spacing.

        Filters 0-20 land on only 1-2 DFT bins each, and exactly one adjacent
        pair (13, 14) ends up sharing no bin at all. This is a genuine property
        of Whisper's own filterbank — we match their array to 1e-9 — and it
        means the bottom ~quarter of the mel channels are aliased copies of a
        handful of DFT bins rather than independent measurements.
        """
        fb = mel.mel_filterbank()
        disjoint = [
            i for i in range(fb.shape[0] - 1)
            if not ((fb[i] > 0) & (fb[i + 1] > 0)).any()
        ]
        assert disjoint == [13]

    def test_low_filters_are_undersampled_at_whisper_settings(self):
        """The consequence of 40 Hz bins with mel bands only ~10 Hz wide."""
        fb = mel.mel_filterbank()
        widths = [(row > 0).sum().item() for row in fb]
        assert max(widths[:20]) <= 2  # bottom bands see 1-2 bins
        assert min(widths[-10:]) >= 8  # top bands see many
        assert all(w > 0 for w in widths)  # but none is empty

    def test_slaney_normalisation_is_applied(self):
        """Equal *area*, not equal peak. If every row peaked at 1.0 we'd have
        skipped the normalisation and would not match Whisper."""
        fb = mel.mel_filterbank()
        peaks = fb.max(dim=1).values
        assert not torch.allclose(peaks, torch.ones_like(peaks), atol=0.05)
        # Low-frequency filters are narrow, so normalising gives them tall peaks.
        assert peaks[0] > peaks[-1] * 5

    def test_low_filters_are_narrower_than_high_ones(self):
        fb = mel.mel_filterbank()
        widths = [(row > 0).sum().item() for row in fb]
        assert widths[0] < widths[-1]
        assert sum(widths[:10]) < sum(widths[-10:])

    def test_covers_the_full_band(self):
        """Every DFT bin above the first should be seen by some filter."""
        fb = mel.mel_filterbank()
        covered = (fb > 0).any(dim=0)
        assert covered[1:-1].float().mean() > 0.98


class TestLogMelSpectrogram:
    def test_output_shape_is_80_by_3000(self):
        wav = audio.sine(440, 5.0)
        assert mel.log_mel_spectrogram(wav).shape == (80, 3000)

    def test_long_audio_is_trimmed_to_3000(self):
        assert mel.log_mel_spectrogram(audio.sine(440, 45.0)).shape == (80, 3000)

    def test_batched_input(self):
        batch = torch.stack([audio.sine(f, 2.0) for f in (200, 400, 800)])
        assert mel.log_mel_spectrogram(batch).shape == (3, 80, 3000)

    def test_unpadded_frame_count(self):
        """10 s at hop 160 -> 1000 frames."""
        assert mel.log_mel_spectrogram(audio.sine(440, 10.0), pad_to=None).shape == (80, 1000)

    def test_range_is_two_wide_but_its_position_depends_on_level(self):
        """The paper says "globally scale the input to be between -1 and 1".

        The span is always exactly 2.0 (the -8 floor divided by 4), but *where*
        that window sits depends on the input's absolute level — the frontend
        has no amplitude normalisation. A full-scale chirp lands at
        [-0.40, 1.60]; the same chirp at 1% amplitude lands at [-1.40, 0.60].
        Typical speech sits near [-1, 1], which is what the paper describes.
        """
        for amp, expected_min in ((1.0, -0.397), (0.1, -0.897), (0.01, -1.397)):
            spec = mel.log_mel_spectrogram(audio.chirp(80, 7000, 20.0, amplitude=amp))
            assert spec.min().item() == pytest.approx(expected_min, abs=0.01)
            assert (spec.max() - spec.min()).item() == pytest.approx(2.0, abs=1e-4)

    def test_dynamic_range_is_exactly_two(self):
        """The -8 floor then /4 makes the span exactly 2.0 for any input."""
        spec = mel.log_mel_spectrogram(audio.chirp(80, 7000, 20.0))
        assert (spec.max() - spec.min()).item() == pytest.approx(2.0, abs=1e-5)

    def test_is_deterministic(self):
        """A fixed affine map, not batch norm: same audio -> same tensor."""
        wav = audio.sine(440, 3.0)
        assert torch.equal(mel.log_mel_spectrogram(wav), mel.log_mel_spectrogram(wav))

    def test_a_tone_lights_up_one_mel_band(self):
        spec = mel.log_mel_spectrogram(audio.sine(1000, 5.0), pad_to=None)
        active = spec[:, 10:-10].mean(dim=1)
        peak_band = int(torch.argmax(active))
        # 1 kHz is the mel breakpoint at 15 mel, out of ~46 mel at Nyquist,
        # so it should land around a third of the way up the 80 bands.
        assert 20 <= peak_band <= 35

    def test_higher_tone_lands_in_a_higher_band(self):
        def band(freq):
            s = mel.log_mel_spectrogram(audio.sine(freq, 5.0), pad_to=None)
            return int(torch.argmax(s[:, 10:-10].mean(dim=1)))

        assert band(200) < band(1000) < band(4000)

    def test_silence_is_not_zero(self):
        """A trap: padding maps to a constant *negative* value, not 0."""
        spec = mel.log_mel_spectrogram(torch.zeros(audio.N_SAMPLES))
        assert torch.allclose(spec, spec[0, 0].expand_as(spec))
        assert spec[0, 0].item() < 0

    def test_the_frontend_is_NOT_loudness_invariant(self):
        """Worth pinning down, because it is easy to assume otherwise.

        Scaling the waveform by k shifts the whole log-mel by log10(k^2)/4 — a
        constant offset, not a no-op. The -8 floor preserves the *shape*, but
        nothing removes the absolute level. So the same sentence recorded loud
        and quiet gives the encoder two different tensors, and robustness to
        that has to be learned from data rather than handed over by the
        frontend. (We match openai-whisper exactly here; this is their
        behaviour, not our deviation.)
        """
        wav = audio.chirp(100, 6000, 10.0, amplitude=0.5)
        loud = mel.log_mel_spectrogram(wav)
        quiet = mel.log_mel_spectrogram(wav * 0.01)

        assert not torch.allclose(loud, quiet, atol=1e-2)

        # Wherever there is real signal, the shift is exactly log10(k^2)/4 = 1.0.
        diff = loud - quiet
        assert diff.max().item() == pytest.approx(1.0, abs=1e-3)

    def test_the_absolute_clamp_overtakes_the_relative_floor_when_quiet(self):
        """Two floors compete, and which one binds depends on input level.

        `clamp(mel, min=1e-10)` is an *absolute* floor at log10 = -10.
        `maximum(x, x.max() - 8)` is a *relative* floor 80 dB below the peak.

        For normal-level audio the relative floor is higher, so it binds and
        silence maps to (peak - 8). But scale the input down ~100x and the peak
        drops until (peak - 8) falls below -10 — at which point the absolute
        clamp binds instead and the relative floor never fires.

        The visible consequence: identical audio at two levels differs by a
        constant 1.0 in the loud regions but 0.95 in the silent ones, so the
        two spectrograms are not even related by a shift. Quiet recordings get
        a genuinely different representation, not just an offset one.
        """
        wav = audio.chirp(100, 6000, 10.0, amplitude=0.5)

        def floor_stats(x):
            spec = mel.log_mel_spectrogram(x)
            log10_peak = spec.max().item() * 4 - 4
            return log10_peak, log10_peak - 8.0

        loud_peak, loud_floor = floor_stats(wav)
        quiet_peak, quiet_floor = floor_stats(wav * 0.01)

        assert loud_peak - quiet_peak == pytest.approx(4.0, abs=1e-3)  # log10(1e-4)
        assert loud_floor > -10.0  # relative floor binds
        assert quiet_floor < -10.0  # absolute clamp binds instead

        diff = mel.log_mel_spectrogram(wav) - mel.log_mel_spectrogram(wav * 0.01)
        offsets = torch.unique(torch.round(diff * 1e4) / 1e4)
        assert offsets.min().item() == pytest.approx(0.9524, abs=1e-3)
        assert offsets.max().item() == pytest.approx(1.0, abs=1e-3)


class TestAgainstReferenceImplementation:
    """End-to-end equality with the openai-whisper package, when available."""

    @pytest.fixture(scope="class")
    def ow(self):
        return pytest.importorskip(
            "whisper",
            reason='install with: uv run --with openai-whisper --with "numba>=0.61" pytest',
        )

    @pytest.mark.parametrize(
        "name",
        ["noise_30s", "tone_3s", "chirp_10s", "quiet"],
    )
    def test_frontend_matches(self, ow, name):
        torch.manual_seed(0)
        wav = {
            "noise_30s": torch.randn(audio.N_SAMPLES) * 0.1,
            "tone_3s": audio.sine(440, 3.0),
            "chirp_10s": audio.chirp(80, 7500, 10.0),
            "quiet": audio.chirp(200, 3000, 5.0) * 0.001,
        }[name]

        ours = mel.log_mel_spectrogram(wav, pad_to=None)
        theirs = ow.log_mel_spectrogram(wav.numpy(), n_mels=80)
        assert ours.shape == theirs.shape
        assert torch.allclose(ours, theirs, atol=1e-5)

    def test_padded_path_matches(self, ow):
        wav = audio.sine(440, 3.0)
        ours = mel.log_mel_spectrogram(wav)
        theirs = ow.log_mel_spectrogram(ow.pad_or_trim(wav.numpy()), n_mels=80)
        assert torch.allclose(ours, theirs, atol=1e-5)

    def test_128_mel_matches(self, ow):
        torch.manual_seed(0)
        wav = torch.randn(audio.N_SAMPLES) * 0.1
        ours = mel.log_mel_spectrogram(wav, n_mels=128)
        theirs = ow.log_mel_spectrogram(wav.numpy(), n_mels=128)
        assert torch.allclose(ours, theirs, atol=1e-5)


def test_frontend_constants_match_the_paper():
    assert mel.N_FFT == 400  # 25 ms
    assert mel.HOP_LENGTH == 160  # 10 ms
    assert mel.N_MELS == 80  # "80-channel log-magnitude Mel spectrogram"
    assert mel.N_FRAMES == 3000
    assert mel.N_FFT / audio.SAMPLE_RATE == pytest.approx(0.025)
    assert mel.HOP_LENGTH / audio.SAMPLE_RATE == pytest.approx(0.010)
