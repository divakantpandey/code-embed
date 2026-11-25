"""Unit tests for training callbacks (AlertCallback, CheckpointCallback, etc.)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.training.callbacks import AlertCallback


class TestAlertCallback:
    """Tests for AlertCallback loss spike, NaN handling, and gradient explosion detection."""

    def test_alert_callback_ignores_tiny_loss_fluctuations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify that relative spikes below min_loss_threshold do not trigger alerts."""
        cb = AlertCallback(
            loss_spike_multiplier=3.0,
            min_loss_threshold=0.01,
            rolling_window_size=10,
        )
        trainer = MagicMock()

        # Seed rolling window with baseline tiny losses (~1e-5)
        with caplog.at_level(logging.WARNING):
            for step in range(1, 6):
                cb.on_step_end(trainer, step, {"loss": 1e-5})

            # Now introduce a 5x spike, but still well below min_loss_threshold (5e-5 < 0.01)
            cb.on_step_end(trainer, 6, {"loss": 5e-5})

        # No warning should be emitted
        assert "ALERT: Loss spike detected!" not in caplog.text

    def test_alert_callback_triggers_on_real_loss_spike(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify that spikes above min_loss_threshold do trigger alerts."""
        cb = AlertCallback(
            loss_spike_multiplier=3.0,
            min_loss_threshold=0.01,
            rolling_window_size=10,
        )
        trainer = MagicMock()

        with caplog.at_level(logging.WARNING):
            for step in range(1, 6):
                cb.on_step_end(trainer, step, {"loss": 0.05})

            # Spike from 0.05 to 0.25 (5x multiplier, and > 0.01)
            cb.on_step_end(trainer, 6, {"loss": 0.25})

        assert "ALERT: Loss spike detected!" in caplog.text
        assert "Step 6" in caplog.text

    def test_alert_callback_nan_halts_trainer(self) -> None:
        """Verify that NaN loss halts the trainer when nan_halt is True."""
        cb = AlertCallback(nan_halt=True)
        trainer = MagicMock()
        trainer.should_stop = False

        cb.on_step_end(trainer, 1, {"loss": float("nan")})

        assert trainer.should_stop is True

    def test_alert_callback_grad_norm_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify that grad norm exceeding threshold logs a warning."""
        cb = AlertCallback(grad_norm_threshold=5.0)
        trainer = MagicMock()

        with caplog.at_level(logging.WARNING):
            cb.on_step_end(trainer, 1, {"loss": 0.1, "grad_norm": 8.5})

        assert "ALERT: Large gradient norm detected!" in caplog.text
        assert "grad_norm=8.4" in caplog.text or "grad_norm=8.5" in caplog.text
