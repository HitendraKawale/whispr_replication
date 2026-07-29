"""Step 2 tests: our from-scratch DFT/STFT must equal PyTorch's FFT-based one.

This is the test that earns us the right to use `torch.stft` in the real
pipeline: if the readable version and the fast version agree, the fast version
is no longer a black box.
"""

from __future__ import annotations

import math

import pytest
import torch

from whispr import audio, dft


class TestDFTMatrix:
    def test_matches_torch_fft(self):
        x = torch.randn(64)
        assert torch.allclose(dft.dft(x), torch.fft.fft(x), atol=1e-4)

    def test_rfft_matches_torch(self):
        x = torch.randn(128)
        assert torch.allclose(dft.rfft_naive(x), torch.fft.rfft(x), atol=1e-4)

    def test_rfft_keeps_half_plus_one_bins(self):
        for n in (64, 128, 400):
            assert dft.rfft_naive(torch.randn(n)).shape[-1] == n // 2 + 1

    def test_whisper_n_fft_gives_201_bins(self):
        """The 201 that the mel filterbank projects down from."""
        assert dft.rfft_naive(torch.randn(400)).shape[-1] == 201

    def test_dft_matrix_is_unitary_up_to_scale(self):
        """W·Wᴴ = N·I — the DFT is a rotation (times sqrt(N))."""
        n = 32
        w = dft.dft_matrix(n)
        prod = w @ w.conj().T
        assert torch.allclose(prod, n * torch.eye(n, dtype=torch.complex64), atol=1e-3)

    def test_dc_bin_is_the_sum(self):
        """Bin 0 probes at frequency 0, so it's just the sum of the signal."""
        x = torch.randn(64)
        assert dft.dft(x)[0].real.item() == pytest.approx(x.sum().item(), abs=1e-3)

    def test_conjugate_symmetry_for_real_input(self):
        """X[N-k] = conj(X[k]) — why we can throw away half the spectrum."""
        x = torch.randn(64)
        full = dft.dft(x)
        assert torch.allclose(full[1:32], full[33:].flip(0).conj(), atol=1e-4)


class TestBinMapping:
    def test_whisper_bin_spacing_is_40hz(self):
        assert dft.bin_to_hz(1, n_fft=400, sample_rate=16_000) == pytest.approx(40.0)

    def test_top_bin_is_nyquist(self):
        assert dft.bin_to_hz(200, n_fft=400, sample_rate=16_000) == pytest.approx(8000.0)

    def test_a_pure_tone_lands_in_the_predicted_bin(self):
        sr, n_fft = 16_000, 400
        freq = 1000.0
        x = audio.sine(freq, n_fft / sr, sr)
        spec = dft.rfft_naive(x * dft.hann_window(n_fft))
        peak = int(torch.argmax(spec.abs()))
        assert dft.bin_to_hz(peak, n_fft, sr) == pytest.approx(freq, abs=40.0)


class TestHannWindow:
    def test_matches_torch_periodic(self):
        assert torch.allclose(dft.hann_window(400), torch.hann_window(400), atol=1e-6)

    def test_matches_torch_symmetric(self):
        assert torch.allclose(
            dft.hann_window(400, periodic=False),
            torch.hann_window(400, periodic=False),
            atol=1e-6,
        )

    def test_starts_at_zero_and_peaks_at_one(self):
        w = dft.hann_window(400)
        assert w[0].item() == pytest.approx(0.0, abs=1e-6)
        assert w.max().item() == pytest.approx(1.0, abs=1e-3)

    def test_periodic_and_symmetric_actually_differ(self):
        """They differ by one sample of stretch — the classic silent bug."""
        assert not torch.allclose(dft.hann_window(64), dft.hann_window(64, periodic=False))

    def test_reduces_spectral_leakage(self):
        """The reason windows exist, as a number.

        Use a frequency deliberately *between* bins (1020 Hz with 40 Hz bins) —
        that's when a rectangular window leaks worst.
        """
        sr, n_fft = 16_000, 400
        x = audio.sine(1020.0, n_fft / sr, sr)

        rect = dft.rfft_naive(x).abs()
        hann = dft.rfft_naive(x * dft.hann_window(n_fft)).abs()

        def sidelobe_ratio(spec):
            peak = int(torch.argmax(spec))
            mask = torch.ones_like(spec, dtype=torch.bool)
            mask[max(0, peak - 4) : peak + 5] = False
            return (spec[mask].max() / spec[peak]).item()

        assert sidelobe_ratio(hann) < sidelobe_ratio(rect) / 5


class TestFraming:
    def test_frame_count_and_shape(self):
        x = torch.arange(1000, dtype=torch.float32)
        f = dft.frame(x, frame_length=400, hop_length=160)
        assert f.shape == (1 + (1000 - 400) // 160, 400)

    def test_frames_overlap_correctly(self):
        x = torch.arange(100, dtype=torch.float32)
        f = dft.frame(x, frame_length=10, hop_length=4)
        assert f[0, 0].item() == 0
        assert f[1, 0].item() == 4  # second frame starts one hop later
        assert torch.equal(f[0, 4:], f[1, :6])  # the overlapping region matches


class TestSTFT:
    @pytest.mark.parametrize("n_fft,hop", [(400, 160), (256, 128), (512, 256)])
    def test_matches_torch_stft(self, n_fft, hop):
        x = torch.randn(16_000)
        ours = dft.stft_naive(x, n_fft, hop, window=dft.hann_window(n_fft), center=True)
        theirs = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            window=torch.hann_window(n_fft),
            center=True,
            return_complex=True,
        )
        assert ours.shape == theirs.shape
        assert torch.allclose(ours, theirs, atol=1e-3)

    def test_matches_torch_stft_uncentered(self):
        x = torch.randn(8000)
        ours = dft.stft_naive(x, 400, 160, window=dft.hann_window(400), center=False)
        theirs = torch.stft(
            x, n_fft=400, hop_length=160, window=torch.hann_window(400),
            center=False, return_complex=True,
        )
        assert torch.allclose(ours, theirs, atol=1e-3)

    def test_whisper_window_gives_3001_frames(self):
        """480,000 samples -> 3001 frames; Whisper drops one to get 3000."""
        x = torch.zeros(audio.N_SAMPLES)
        spec = dft.stft_naive(x, 400, 160, center=True)
        assert spec.shape == (201, 3001)
        assert spec[:, :-1].shape[1] == 3000

    def test_spectrogram_of_a_tone_is_a_horizontal_line(self):
        """Sanity: a constant tone should be constant across time."""
        sr = 16_000
        x = audio.sine(1000, 1.0, sr)
        mag = dft.stft_naive(x, 400, 160).abs()
        interior = mag[:, 5:-5]  # avoid reflect-padding edge effects
        peak_bins = interior.argmax(dim=0)
        assert peak_bins.float().std().item() < 0.1  # same bin every frame
        assert dft.bin_to_hz(int(peak_bins[0]), 400, sr) == pytest.approx(1000, abs=40)


class TestResolutionTradeoff:
    def test_frequency_resolution_improves_with_n_fft(self):
        sr = 16_000
        for n_fft in (128, 400, 2048):
            assert dft.bin_to_hz(1, n_fft, sr) == pytest.approx(sr / n_fft)
        # And the product of the two resolutions is always 1 — uncertainty.
        for n_fft in (128, 400, 2048):
            freq_res = sr / n_fft
            time_res = n_fft / sr
            assert freq_res * time_res == pytest.approx(1.0)

    def test_long_window_resolves_two_close_tones_short_one_cannot(self):
        sr = 16_000
        x = audio.sine(1000, 0.5, sr) + audio.sine(1080, 0.5, sr)  # 80 Hz apart

        def peak_count(n_fft):
            seg = x[:n_fft] * dft.hann_window(n_fft)
            m = dft.rfft_naive(seg).abs()
            # Band must be wide enough to have an interior at n_fft=128, where
            # bins are 125 Hz apart.
            lo, hi = int(500 * n_fft / sr), int(1600 * n_fft / sr)
            band = m[lo:hi]
            # count local maxima above half the band's peak
            thr = band.max() * 0.5
            return sum(
                1
                for i in range(1, len(band) - 1)
                if band[i] > thr and band[i] >= band[i - 1] and band[i] > band[i + 1]
            )

        assert peak_count(128) == 1  # 125 Hz bins: the two tones merge
        assert peak_count(2048) == 2  # 7.8 Hz bins: cleanly separated


def test_power_and_magnitude_relationship():
    x = torch.randn(1000)
    spec = dft.stft_naive(x, 256, 128)
    assert torch.allclose(dft.power(spec), dft.magnitude(spec) ** 2, atol=1e-4)


def test_parseval_energy_is_conserved():
    """Time-domain energy == frequency-domain energy / N. The DFT is a rotation."""
    x = torch.randn(256)
    time_energy = (x**2).sum().item()
    freq_energy = (dft.dft(x).abs() ** 2).sum().item() / 256
    assert freq_energy == pytest.approx(time_energy, rel=1e-3)


def test_dft_matrix_first_row_is_all_ones():
    """Row 0 probes frequency 0: exp(0) = 1 everywhere."""
    w = dft.dft_matrix(16)
    assert torch.allclose(w[0].real, torch.ones(16), atol=1e-6)
    assert torch.allclose(w[0].imag, torch.zeros(16), atol=1e-6)


def test_shifting_a_signal_changes_phase_not_magnitude():
    """The shift theorem — and why magnitude-only is time-shift robust."""
    x = audio.sine(500, 400 / 16_000, 16_000)
    shifted = torch.roll(x, 37)
    assert torch.allclose(dft.dft(x).abs(), dft.dft(shifted).abs(), atol=1e-2)
    assert not torch.allclose(dft.dft(x).angle(), dft.dft(shifted).angle(), atol=1e-2)


def test_griffin_lim_note_mentions_phase():
    assert "Phase" in dft.griffin_lim_note()
    assert math.isfinite(1.0)
