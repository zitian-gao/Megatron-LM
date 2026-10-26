# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.transformer.moe.moe_logging import (
    MoEMetricsTracker,
    _format_router_load_balancing_metric_name,
    _parse_router_load_balancing_metric_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("expert_ratio", ("expert_ratio", None, None)),
        ("drop_ratio_layer_loop_4", ("drop_ratio", None, 4)),
        ("maxvio_model_loop_2", ("maxvio", 2, None)),
        ("avgvio_model_loop_3_layer_loop_5", ("avgvio", 3, 5)),
    ],
)
def test_parse_router_load_balancing_metric_suffix(name, expected):
    assert _parse_router_load_balancing_metric_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "unknown_layer_loop_1",
        "expert_ratio_model_loop_x",
        "drop_ratio_layer_loop_2_model_loop_1",
        "maxvio_model_loop_1_layer_loop_2_extra",
    ],
)
def test_parse_router_load_balancing_metric_rejects_invalid_suffix(name):
    assert _parse_router_load_balancing_metric_name(name) is None


def test_format_router_load_balancing_metric_suffix():
    name = "avgvio_model_loop_3_layer_loop_5"

    assert _format_router_load_balancing_metric_name(name) == "avgvio_model_loop3_loop5"
    assert (
        _format_router_load_balancing_metric_name(name, layer_idx=1)
        == "avgvio_layer2_model_loop3_loop5"
    )
    assert (
        _format_router_load_balancing_metric_name(name, layer_idx=1, wandb_section=True)
        == "load_balancing/avgvio_layer2_model_loop3_loop5"
    )
    assert _format_router_load_balancing_metric_name("uelb_load_balancing_loss") is None


def test_router_metric_report_logs_tb_and_wandb_summary_and_layers_by_default():
    tracker = MoEMetricsTracker()
    metric_name = "expert_ratio_model_loop_2_layer_loop_3"
    tracker.record(metric_name, torch.tensor(2.0), layer_number=1, num_layers=2)
    tracker.record(metric_name, torch.tensor(4.0), layer_number=2, num_layers=2)
    writer = Mock()
    wandb_writer = Mock()

    with patch.object(tracker, "_sync_metrics") as sync_metrics:
        tracker.report(
            loss_scale=1.0,
            iteration=11,
            writer=writer,
            wandb_writer=wandb_writer,
            per_layer_logging=False,
            track_names=[metric_name],
            num_layers=2,
        )

    sync_metrics.assert_called_once_with([metric_name], None)

    tb_values = {}
    for call in writer.add_scalar.call_args_list:
        key, value, iteration = call.args
        assert iteration == 11
        tb_values[key] = float(value)

    assert tb_values == pytest.approx(
        {
            "moe/load_balancing/expert_ratio_model_loop2_loop3": 3.0,
            "moe/load_balancing/expert_ratio_layer1_model_loop2_loop3": 2.0,
            "moe/load_balancing/expert_ratio_layer2_model_loop2_loop3": 4.0,
        }
    )

    wandb_values = {}
    for call in wandb_writer.log.call_args_list:
        values, iteration = call.args
        assert iteration == 11
        wandb_values.update({key: float(value) for key, value in values.items()})

    assert wandb_values == pytest.approx(
        {
            "load_balancing/expert_ratio_model_loop2_loop3": 3.0,
            "load_balancing/expert_ratio_layer1_model_loop2_loop3": 2.0,
            "load_balancing/expert_ratio_layer2_model_loop2_loop3": 4.0,
        }
    )
