"""Tests for streaming_eeg/nodes.py — no Spark required, pure Python/Pandas."""
import numpy as np
import pandas as pd
import pytest

from neuro_spark.core import BANDS, MOTOR_CHANNELS
from neuro_spark.pipelines.streaming_eeg.nodes import (
    FEATURE_ORDER,
    _band_power,
    compute_alpha_topomap,
    compute_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

FS = 160
EPOCH_SAMPLES = FS * 2  # 2-second epoch


def _make_epoch(task_label: str = "T1", seed: int = 0) -> pd.DataFrame:
    """Minimal epoch DataFrame with all MOTOR_CHANNELS and required meta columns."""
    rng = np.random.default_rng(seed)
    n = EPOCH_SAMPLES
    data = {ch: rng.standard_normal(n) for ch in MOTOR_CHANNELS}
    data["time"] = np.linspace(0, 2, n, endpoint=False)
    data["epoch_id"] = 0
    data["epoch_label"] = task_label
    data["task_label"] = task_label
    data["subject_id"] = "S001"
    data["run_id"] = "R04"
    data["event_time"] = "2026-01-01T00:00:00"
    data["run_seq"] = 0
    return pd.DataFrame(data)


# ── _band_power ───────────────────────────────────────────────────────────────

class TestBandPower:
    def test_pure_sine_in_band(self):
        """A 10 Hz sine should have all power in the alpha band (8–13 Hz)."""
        t = np.linspace(0, 2, EPOCH_SAMPLES, endpoint=False)
        sig = np.sin(2 * np.pi * 10 * t)
        lo, hi = BANDS["alpha"]
        alpha = _band_power(sig, lo, hi)
        theta = _band_power(sig, *BANDS["theta"])
        assert alpha > theta * 100, "Alpha power should dominate for a 10 Hz sine"

    def test_pure_sine_out_of_band(self):
        """A 10 Hz sine should have near-zero power in the gamma band."""
        t = np.linspace(0, 2, EPOCH_SAMPLES, endpoint=False)
        sig = np.sin(2 * np.pi * 10 * t)
        gamma = _band_power(sig, *BANDS["gamma"])
        assert gamma < 1e-6

    def test_short_signal_returns_zero(self):
        assert _band_power(np.array([1.0, 2.0]), 8, 13) == 0.0

    def test_empty_signal_returns_zero(self):
        assert _band_power(np.array([]), 8, 13) == 0.0

    def test_returns_float(self):
        sig = np.random.default_rng(42).standard_normal(EPOCH_SAMPLES)
        result = _band_power(sig, 8, 13)
        assert isinstance(result, float)


# ── compute_features ──────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_returns_all_expected_keys(self):
        epoch = _make_epoch()
        feats = compute_features(epoch)
        assert set(feats.keys()) == set(FEATURE_ORDER)

    def test_feature_order_consistent(self):
        """FEATURE_ORDER must match MOTOR_CHANNELS × BANDS + diff bands."""
        expected = (
            [f"{ch}_{band}" for ch in MOTOR_CHANNELS for band in BANDS]
            + [f"diff_{band}" for band in BANDS]
        )
        assert FEATURE_ORDER == expected

    def test_lateralization_correct(self):
        """diff_alpha = C3_alpha - C4_alpha by definition."""
        epoch = _make_epoch(seed=7)
        feats = compute_features(epoch)
        assert pytest.approx(feats["diff_alpha"]) == feats["C3_alpha"] - feats["C4_alpha"]

    def test_all_values_are_floats(self):
        feats = compute_features(_make_epoch())
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} is not a float"

    def test_all_values_non_negative(self):
        """Band power is always >= 0; diff can be negative."""
        feats = compute_features(_make_epoch())
        for band in BANDS:
            for ch in MOTOR_CHANNELS:
                assert feats[f"{ch}_{band}"] >= 0.0

    def test_different_seeds_give_different_features(self):
        f1 = compute_features(_make_epoch(seed=1))
        f2 = compute_features(_make_epoch(seed=2))
        assert f1 != f2

    def test_uses_core_motor_channels(self):
        """No dots in channel names — must match core.MOTOR_CHANNELS."""
        for ch in MOTOR_CHANNELS:
            assert "." not in ch, f"Channel '{ch}' has dots — should be clean"


# ── compute_alpha_topomap ─────────────────────────────────────────────────────

class TestComputeAlphaTopomap:
    def test_excludes_meta_columns(self):
        epoch = _make_epoch()
        topo = compute_alpha_topomap(epoch)
        meta = {"subject_id", "run_id", "time", "task_label", "epoch_id",
                "epoch_label", "event_time", "run_seq"}
        assert not meta.intersection(topo.keys())

    def test_includes_motor_channels(self):
        epoch = _make_epoch()
        topo = compute_alpha_topomap(epoch)
        for ch in MOTOR_CHANNELS:
            assert ch in topo, f"Motor channel {ch} missing from topomap"

    def test_values_non_negative(self):
        topo = compute_alpha_topomap(_make_epoch())
        for ch, pw in topo.items():
            assert pw >= 0.0, f"{ch} has negative alpha power"

    def test_values_are_floats(self):
        topo = compute_alpha_topomap(_make_epoch())
        for ch, pw in topo.items():
            assert isinstance(pw, float), f"{ch} value is not a float"

    def test_pure_alpha_sine_dominates(self):
        """A 10 Hz sine on C3 should give high alpha power for that channel."""
        epoch = _make_epoch()
        t = np.linspace(0, 2, EPOCH_SAMPLES, endpoint=False)
        epoch["C3"] = np.sin(2 * np.pi * 10 * t)
        epoch["C4"] = np.zeros(EPOCH_SAMPLES)
        topo = compute_alpha_topomap(epoch)
        assert topo["C3"] > topo["C4"] * 100
