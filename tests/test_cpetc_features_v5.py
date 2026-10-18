from __future__ import annotations

import math

import torch
import numpy as np
import pytest

from mate_spc_cpetc.features import (
    AETrainingConfig,
    ChannelDenoisingAE,
    ChannelNormalizer,
    LoopReferenceState,
    channel_features,
    extract_inclusive_windows,
    extract_window_features,
    fit_loop_reference,
    loop_features,
    split_prefix_monitor,
    train_denoising_ae,
    _one_sided_psd,
    _welch_coherence,
)


def test_inclusive_windows_and_frozen_split_are_exact() -> None:
    run = torch.zeros((1, 500, 52), dtype=torch.float64)
    run[0, :, 0] = torch.arange(500, dtype=torch.float64)
    windows = extract_inclusive_windows(run)
    assert windows.shape == (1, 93, 52, 128)
    torch.testing.assert_close(windows[0, 0, 0], torch.arange(1, 129, dtype=torch.float64))
    torch.testing.assert_close(windows[0, -1, 0], torch.arange(369, 497, dtype=torch.float64))
    prefix, monitor = split_prefix_monitor(windows)
    assert prefix.shape == (1, 8, 52, 128)
    assert monitor.shape == (1, 85, 52, 128)


def test_future_sample_cannot_change_an_earlier_window() -> None:
    baseline = torch.randn((1, 500, 52), generator=torch.Generator().manual_seed(7))
    changed = baseline.clone()
    changed[:, 129:, :] += 100.0
    first = extract_inclusive_windows(baseline)
    second = extract_inclusive_windows(changed)
    torch.testing.assert_close(first[:, 0], second[:, 0])
    assert not torch.equal(first[:, 1], second[:, 1])


def test_normalizer_uses_mad_consistency_and_clips() -> None:
    values = torch.zeros((1, 500, 52), dtype=torch.float64)
    values[0, :, 0] = torch.arange(500, dtype=torch.float64)
    normalizer = ChannelNormalizer().fit(values)
    expected_median = 249.5
    expected_mad = 125.0
    assert float(normalizer.median[0]) == expected_median
    assert float(normalizer.scale[0]) == pytest.approx(1.4826 * expected_mad)
    transformed = normalizer.transform_runs(values)
    assert float(torch.max(torch.abs(transformed))) <= 8.0
    assert float(normalizer.scale[1]) == 1.0e-6


def test_grouped_ae_shape_and_channel_isolation() -> None:
    torch.manual_seed(11)
    model = ChannelDenoisingAE().double().eval()
    baseline = torch.zeros((2, 52, 128), dtype=torch.float64)
    changed = baseline.clone()
    changed[:, 7, 20] = 10.0
    rebuilt0, code0 = model(baseline)
    rebuilt1, code1 = model(changed)
    assert rebuilt0.shape == changed.shape
    assert code0.shape == (2, 52, 2, 32)
    affected = torch.max(torch.abs(rebuilt1 - rebuilt0), dim=-1).values
    assert float(affected[:, 7].max().detach()) > 0.0
    torch.testing.assert_close(affected[:, :7], torch.zeros_like(affected[:, :7]))
    torch.testing.assert_close(affected[:, 8:], torch.zeros_like(affected[:, 8:]))
    torch.testing.assert_close(code0[:, :7], code1[:, :7])


def test_channel_psd_sine_band_and_constant_are_finite() -> None:
    time = torch.arange(128, dtype=torch.float64)
    windows = torch.zeros((2, 52, 128), dtype=torch.float64)
    windows[0, 0] = torch.sin(2.0 * math.pi * 10.0 * time / 128.0)
    model = ChannelDenoisingAE().double().eval()
    reconstruction, code = model(windows)
    values = channel_features(windows, reconstruction, code)
    assert values.shape == (2, 52, 14)
    assert torch.isfinite(values).all()
    six_bands = values[0, 0, 6:12]
    assert int(torch.argmax(six_bands)) == 3  # FFT bin 10 lies in [8,16).
    assert float(values[1, 0, 12].detach()) == pytest.approx(0.0)
    assert float(values[1, 0, 13].detach()) == pytest.approx(0.0)


def test_channel_psd_scaling_entropy_and_centroid_match_numpy() -> None:
    sample = np.random.default_rng(7321).normal(size=128)
    windows = torch.zeros((1, 52, 128), dtype=torch.float64)
    windows[0, 0] = torch.from_numpy(sample)
    observed_power = _one_sided_psd(windows)[0, 0].numpy()

    centered = sample - np.mean(sample)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(128) / 128.0)
    expected_power = np.abs(np.fft.rfft(centered * hann)) ** 2 / np.sum(hann**2)
    expected_power[0] = 0.0
    expected_power[1:64] *= 2.0
    np.testing.assert_allclose(observed_power, expected_power, rtol=1.0e-12, atol=1.0e-12)

    features = channel_features(
        windows,
        torch.zeros_like(windows),
        torch.zeros((1, 52, 2, 32), dtype=torch.float64),
    )[0, 0]
    total = max(float(np.sum(expected_power[1:65])), 1.0e-12)
    q = expected_power[1:65] / total
    bands = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 65))
    expected_spectral = [
        np.log(np.sum(expected_power[start:stop]) / total + 1.0e-12)
        for start, stop in bands
    ]
    expected_spectral.extend((
        -np.sum(q * np.log(q + 1.0e-12)) / np.log(64.0),
        np.sum(q * (np.arange(1, 65) / 64.0)),
    ))
    np.testing.assert_allclose(
        features[6:14].numpy(), np.asarray(expected_spectral), rtol=1.0e-12, atol=1.0e-12
    )


def test_loop_reference_and_stiction_features_are_finite() -> None:
    generator = torch.Generator().manual_seed(91)
    healthy = 0.2 * torch.randn((5, 500, 52), generator=generator, dtype=torch.float64)
    healthy[:, :, 10] += 0.6 * healthy[:, :, 51]
    healthy[:, :, 21] -= 0.3 * healthy[:, :, 51]
    reference = fit_loop_reference(healthy)
    assert reference.dwell_threshold >= 1.0e-6
    windows = torch.zeros((2, 52, 128), dtype=torch.float64)
    windows[0] = healthy[0, 1:129].T
    windows[1] = windows[0]
    windows[1, 51, :] = 0.0
    windows[1, 51, 96:] = 3.0
    values = loop_features(windows, reference)
    assert values.shape == (2, 12)
    assert torch.isfinite(values).all()
    assert float(values[1, 0]) > float(values[0, 0])
    assert float(values[1, 1]) > 0.0


def test_loop_reference_matches_numpy_linear_quantile_direct_solve_and_93_slices() -> None:
    rng = np.random.default_rng(20260912)
    healthy = rng.normal(scale=0.15, size=(4, 500, 52)).astype(np.float64)
    u = healthy[:, :, 51]
    for channel, coefficients, noise_scale in (
        (10, (0.04, 0.72, 0.31), 0.025),
        (21, (-0.03, 0.51, -0.27), 0.035),
    ):
        y = np.empty((4, 500), dtype=np.float64)
        y[:, 0] = rng.normal(scale=0.1, size=4)
        noise = rng.normal(scale=noise_scale, size=(4, 499))
        for time_index in range(1, 500):
            y[:, time_index] = (
                coefficients[0]
                + coefficients[1] * y[:, time_index - 1]
                + coefficients[2] * u[:, time_index - 1]
                + noise[:, time_index - 1]
            )
        healthy[:, :, channel] = y

    observed = fit_loop_reference(torch.from_numpy(healthy))
    expected_tau = max(
        float(np.quantile(np.abs(np.diff(u, axis=1)).reshape(-1), 0.20, method="linear")),
        1.0e-6,
    )
    assert observed.dwell_threshold == expected_tau

    endpoints = tuple(range(128, 497, 4))
    assert len(endpoints) == 93
    for response_index, channel in enumerate((10, 21)):
        y = healthy[:, :, channel]
        design = np.stack(
            (
                np.ones_like(y[:, :-1]),
                y[:, :-1],
                u[:, :-1],
            ),
            axis=-1,
        ).reshape(-1, 3)
        target = y[:, 1:].reshape(-1)
        gram = design.T @ design + np.diag((0.0, 1.0e-4, 1.0e-4))
        beta = np.linalg.solve(gram, design.T @ target)
        residual = y[:, 1:] - (
            beta[0] + beta[1] * y[:, :-1] + beta[2] * u[:, :-1]
        )
        window_mse = np.stack(
            [np.mean(residual[:, endpoint - 127 : endpoint] ** 2, axis=1) for endpoint in endpoints],
            axis=1,
        )
        assert window_mse.shape == (4, 93)
        expected_denominator = max(float(np.quantile(window_mse.reshape(-1), 0.5)), 1.0e-8)
        np.testing.assert_allclose(
            np.asarray(observed.arx_coefficients[response_index]),
            beta,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        assert observed.residual_denominators[response_index] == pytest.approx(
            expected_denominator,
            rel=2.0e-13,
            abs=2.0e-15,
        )


def test_welch_coherence_matches_frozen_one_sided_periodogram() -> None:
    generator = torch.Generator().manual_seed(20260912)
    regular = torch.randn((3, 128), generator=generator, dtype=torch.float64)
    tiny = 1.0e-7 * torch.randn((2, 128), generator=generator, dtype=torch.float64)
    u = torch.cat((regular, tiny), dim=0)
    y = 0.35 * u + torch.cat((
        torch.randn((3, 128), generator=generator, dtype=torch.float64),
        1.0e-7 * torch.randn((2, 128), generator=generator, dtype=torch.float64),
    ), dim=0)
    observed = _welch_coherence(u, y).numpy()

    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(32) / 32.0)
    energy = np.sum(hann**2)
    expected = []
    for current_u, current_y in zip(u.numpy(), y.numpy(), strict=True):
        u_segments = np.stack([current_u[start : start + 32] for start in range(0, 97, 16)])
        y_segments = np.stack([current_y[start : start + 32] for start in range(0, 97, 16)])
        u_segments -= np.mean(u_segments, axis=-1, keepdims=True)
        y_segments -= np.mean(y_segments, axis=-1, keepdims=True)
        uf = np.fft.rfft(u_segments * hann, axis=-1)
        yf = np.fft.rfft(y_segments * hann, axis=-1)
        sxx = np.mean(np.abs(uf) ** 2 / energy, axis=0)
        syy = np.mean(np.abs(yf) ** 2 / energy, axis=0)
        sxy = np.mean(uf * np.conj(yf) / energy, axis=0)
        sxx[1:16] *= 2.0
        syy[1:16] *= 2.0
        sxy[1:16] *= 2.0
        coherence = np.clip(np.abs(sxy) ** 2 / (sxx * syy + 1.0e-12), 0.0, 1.0)
        expected.append(
            np.sum(coherence[1:17] * sxx[1:17])
            / (np.sum(sxx[1:17]) + 1.0e-12)
        )
    np.testing.assert_allclose(observed, np.asarray(expected), rtol=1.0e-12, atol=1.0e-12)


def test_all_loop_features_match_frozen_scalar_definitions_and_order() -> None:
    rng = np.random.default_rng(4109)
    u = np.cumsum(rng.normal(scale=0.035, size=128))
    u[24:31] = u[23]
    u[72:79] = u[71]
    y1 = 0.55 * np.roll(u, 2) + rng.normal(scale=0.025, size=128)
    y2 = -0.35 * np.roll(u, 3) + rng.normal(scale=0.04, size=128)
    windows = torch.zeros((1, 52, 128), dtype=torch.float64)
    windows[0, 51] = torch.from_numpy(u)
    windows[0, 10] = torch.from_numpy(y1)
    windows[0, 21] = torch.from_numpy(y2)
    state = LoopReferenceState(
        0.03,
        ((0.05, 0.60, 0.25), (-0.08, 0.45, -0.20)),
        (0.12, 0.27),
    )
    observed = loop_features(windows, state)[0].numpy()

    du = np.diff(u)
    absolute = np.abs(du)
    expected = [
        np.mean(absolute <= state.dwell_threshold),
        sum(
            bool(np.all(absolute[index - 4 : index] <= state.dwell_threshold))
            * bool(absolute[index] > state.dwell_threshold)
            * np.log1p((absolute[index] / state.dwell_threshold) ** 2)
            for index in range(4, 127)
        ) / 124.0,
        np.mean(du[1:] * du[:-1] < 0.0),
    ]

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        left0 = left - np.mean(left)
        right0 = right - np.mean(right)
        denominator = max(
            float(np.sqrt(np.mean(left0**2) * np.mean(right0**2))), 1.0e-8
        )
        return float(np.mean(left0 * right0) / denominator)

    for response_index, y in enumerate((y1, y2)):
        area = np.sum(0.5 * (u[1:] + u[:-1]) * np.diff(y))
        area /= 127.0 * (np.std(u, ddof=0) * np.std(y, ddof=0) + 1.0e-8)
        forward = max(abs(correlation(u[:-lag], y[lag:])) for lag in range(1, 9))
        reverse = max(abs(correlation(y[:-lag], u[lag:])) for lag in range(1, 9))
        beta = state.arx_coefficients[response_index]
        residual = y[1:] - (beta[0] + beta[1] * y[:-1] + beta[2] * u[:-1])
        ratio = np.log1p(np.mean(residual**2) / state.residual_denominators[response_index])
        coherence = float(
            _welch_coherence(
                torch.from_numpy(u[None, :]), torch.from_numpy(y[None, :])
            )[0]
        )
        expected.extend((area, forward - reverse, ratio, coherence))

    centered = u - np.mean(u)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(128) / 128.0)
    power = np.abs(np.fft.rfft(centered * hann)) ** 2 / np.sum(hann**2)
    power[0] = 0.0
    power[1:64] *= 2.0
    expected.append(np.max(power[1:65]) / max(np.sum(power[1:65]), 1.0e-12))
    np.testing.assert_allclose(observed, np.asarray(expected), rtol=1.0e-11, atol=1.0e-12)


def test_short_ae_fit_freezes_parameters_and_full_feature_shapes() -> None:
    generator = torch.Generator().manual_seed(123)
    raw_runs = torch.randn((2, 500, 52), generator=generator)
    normalizer = ChannelNormalizer().fit(raw_runs)
    normalized = normalizer.transform_runs(raw_runs)
    reference = fit_loop_reference(normalized)
    windows = extract_inclusive_windows(raw_runs)
    autoencoder = ChannelDenoisingAE()
    history = train_denoising_ae(
        autoencoder,
        extract_inclusive_windows(normalized)[:, :2],
        config=AETrainingConfig(epochs=1, batch_size=4),
    )
    assert len(history) == 1 and math.isfinite(history[0])
    assert all(not parameter.requires_grad for parameter in autoencoder.parameters())
    channel_values, loop_values = extract_window_features(
        windows[:, :3],
        normalizer=normalizer,
        autoencoder=autoencoder,
        loop_reference=reference,
        batch_size=2,
    )
    assert channel_values.shape == (2, 3, 52, 14)
    assert loop_values.shape == (2, 3, 12)
    assert torch.isfinite(channel_values).all() and torch.isfinite(loop_values).all()


def test_loop_state_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        LoopReferenceState(float("nan"), ((0.0, 0.0, 0.0),) * 2, (1.0, 1.0))
