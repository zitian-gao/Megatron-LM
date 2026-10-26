# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.training.training import _get_moe_metric_track_names


def test_moe_metric_track_names_include_uelb_and_all_loop_combinations():
    args = SimpleNamespace(
        moe_router_load_balancing_type=["uelb_loss"],
        moe_z_loss_coeff=None,
        layer_loop_times=3,
        model_loop_times=2,
    )

    track_names = _get_moe_metric_track_names(args)
    router_metrics = {"expert_ratio", "drop_ratio", "maxvio", "avgvio"}
    loop_metrics = {
        f"{metric}_model_loop_{model_loop}_layer_loop_{layer_loop}"
        for metric in router_metrics
        for model_loop in range(1, 3)
        for layer_loop in range(1, 4)
    }

    assert set(track_names) == {
        "uelb_load_balancing_loss",
        *router_metrics,
        *loop_metrics,
    }
    assert (
        len(loop_metrics)
        == len(router_metrics) * args.model_loop_times * args.layer_loop_times
    )
