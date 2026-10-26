# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from packaging import version
from pytest import approx
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.test_utilities import Utils


class TestGPTModelLooping:

    def test_model_loop_chains_decoder_hidden_states_and_router_state(self):
        """Pre/post-processing happens once around a chained sequence of decoder calls."""

        class RecordingRouter:
            def __init__(self):
                self.states = []

            def set_model_loop_state(self, loop_idx, model_loop_times):
                self.states.append((loop_idx, model_loop_times))

        class RecordingDecoder:
            def __init__(self, layer):
                self.layers = [layer]
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs.copy())
                return kwargs["hidden_states"] + 1

        router = RecordingRouter()
        layer = SimpleNamespace(is_moe_layer=True, mlp=SimpleNamespace(router=router))
        decoder = RecordingDecoder(layer)
        preprocess_calls = []
        postprocess_calls = []

        rotary_pos_emb = object()
        rotary_pos_cos = object()
        rotary_pos_sin = object()
        rotary_pos_cos_sin = object()
        sequence_len_offset = object()
        padding_mask = object()

        def preprocess(**kwargs):
            preprocess_calls.append(kwargs.copy())
            return (
                kwargs["decoder_input"],
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
                padding_mask,
                rotary_pos_cos_sin,
            )

        def postprocess(**kwargs):
            postprocess_calls.append(kwargs.copy())
            return kwargs["hidden_states"]

        model = SimpleNamespace(
            config=SimpleNamespace(
                fine_grained_activation_offloading=False,
                moe_paged_stash=False,
                model_loop_times=3,
            ),
            decoder=decoder,
            mtp_process=False,
            _preprocess=preprocess,
            _postprocess=postprocess,
        )

        input_ids = object()
        position_ids = object()
        attention_mask = object()
        inference_context = object()
        packed_seq_params = object()
        block_option = object()
        extra_block_kwargs = {"block_option": block_option}

        output = GPTModel.forward(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=torch.tensor(0),
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            padding_mask=padding_mask,
        )

        assert output.item() == 3
        assert len(preprocess_calls) == 1
        assert len(postprocess_calls) == 1
        assert [call["hidden_states"].item() for call in decoder.calls] == [0, 1, 2]
        assert router.states == [(0, 3), (1, 3), (2, 3)]

        expected_decoder_arguments = {
            "attention_mask": attention_mask,
            "inference_context": inference_context,
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "rotary_pos_cos_sin": rotary_pos_cos_sin,
            "packed_seq_params": packed_seq_params,
            "sequence_len_offset": sequence_len_offset,
            "padding_mask": padding_mask,
            "block_option": block_option,
        }
        for call in decoder.calls:
            for name, expected in expected_decoder_arguments.items():
                assert call[name] is expected

        assert preprocess_calls[0]["padding_mask"] is padding_mask
        assert postprocess_calls[0]["hidden_states"].item() == 3
        assert postprocess_calls[0]["decoder_input"].item() == 0
        assert postprocess_calls[0]["extra_block_kwargs"] is extra_block_kwargs
        assert postprocess_calls[0]["inference_context"] is inference_context

    def test_combined_layer_and_model_loops_form_cartesian_state_sequence(self):
        """M model loops each execute all L layer loops with the expected router state."""

        class RecordingRouter:
            def __init__(self):
                self.model_loop_idx = None
                self.layer_loop_idx = None
                self.model_states = []
                self.layer_states = []

            def set_model_loop_state(self, loop_idx, model_loop_times):
                self.model_loop_idx = loop_idx
                self.model_states.append((loop_idx, model_loop_times))

            def set_layer_loop_state(self, loop_idx, layer_loop_times):
                self.layer_loop_idx = loop_idx
                self.layer_states.append((loop_idx, layer_loop_times))

        router = RecordingRouter()
        computation_states = []

        def forward_attention(*args, **kwargs):
            hidden_states = kwargs["hidden_states"] if kwargs else args[0]
            context = kwargs.get("context") if kwargs else args[2]
            computation_states.append((router.model_loop_idx, router.layer_loop_idx))
            return hidden_states + 1, context

        def forward_mlp(
            hidden_states, inference_context, *, padding_mask, packed_seq_params
        ):
            return hidden_states + 1

        layer = SimpleNamespace(
            config=SimpleNamespace(layer_loop_times=2),
            is_moe_layer=True,
            mlp=SimpleNamespace(router=router),
            _forward_attention=forward_attention,
            _forward_mlp=forward_mlp,
        )

        class LayerBackedDecoder:
            def __init__(self):
                self.layers = [layer]
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                hidden_states, _ = TransformerLayer.forward(layer, **kwargs)
                return hidden_states

        decoder = LayerBackedDecoder()
        postprocess_calls = []

        def preprocess(**kwargs):
            return (kwargs["decoder_input"], None, None, None, None, kwargs["padding_mask"])

        def postprocess(**kwargs):
            postprocess_calls.append(kwargs.copy())
            return kwargs["hidden_states"]

        model = SimpleNamespace(
            config=SimpleNamespace(
                fine_grained_activation_offloading=False,
                moe_paged_stash=False,
                model_loop_times=3,
            ),
            decoder=decoder,
            mtp_process=False,
            _preprocess=preprocess,
            _postprocess=postprocess,
        )

        output = GPTModel.forward(
            model,
            input_ids=object(),
            position_ids=object(),
            attention_mask=object(),
            decoder_input=torch.tensor(0),
        )

        assert output.item() == 12
        assert decoder.calls == 3
        assert router.model_states == [(0, 3), (1, 3), (2, 3)]
        assert router.layer_states == [(0, 2), (1, 2)] * 3
        assert computation_states == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        assert len(postprocess_calls) == 1
        assert postprocess_calls[0]["hidden_states"].item() == 12


class TestGPTModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            embedding_init_method_std=1.0,  # Test that we can initialize the embedding weights to something else.
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.gpt_model, GPTModel)

        assert self.gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.gpt_model.parameters()])
        assert num_weights == 6240

    @pytest.mark.internal
    def test_set_input_tensor(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.gpt_model.set_input_tensor(input_tensor)

        assert self.gpt_model.decoder.input_tensor.shape[0] == sequence_length
        assert self.gpt_model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.gpt_model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_embedding_init(self):
        """Test that we can initialize the embedding weights to something else. This test could be added to any model."""
        config: TransformerConfig = self.gpt_model.config
        assert self.gpt_model.embedding.word_embeddings.weight.std().cpu().item() == approx(
            config.embedding_init_method_std, abs=1e-1
        )
        assert self.gpt_model.embedding.word_embeddings.weight.mean().cpu().item() == approx(
            0.0, abs=1e-1
        )

    @pytest.mark.internal
    def test_post_process_forward(self):
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

    @pytest.mark.internal
    def test_output_processor_forward(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()
        self.gpt_model.eval()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = (input_ids + 1) % self.gpt_model.vocab_size
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        context = {"selected_token_positions": torch.tensor([0, 2], device="cuda")}
        seen = {}

        def output_processor(**kwargs):
            seen.update(kwargs)
            logits, _ = kwargs["output_layer"](
                kwargs["hidden_states"],
                weight=kwargs["output_weight"],
                runtime_gather_output=kwargs["runtime_gather_output"],
            )
            logits = kwargs["scale_logits"](logits).transpose(0, 1).contiguous().float()
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                dim=-1, index=kwargs["labels"].unsqueeze(-1)
            )
            token_logprobs = token_logprobs.squeeze(-1)
            return token_logprobs.index_select(1, kwargs["context"]["selected_token_positions"])

        with torch.no_grad():
            logits = self.gpt_model.forward(
                input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
            ).float()
            expected = F.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1))
            expected = expected.squeeze(-1).index_select(1, context["selected_token_positions"])

            output = self.gpt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_processor=output_processor,
                output_processor_context=context,
            )

        assert torch.allclose(output, expected)
        assert seen["context"] is context
        assert seen["output_layer"] is self.gpt_model.output_layer
        assert seen["output_weight"] is None
        assert seen["labels"] is labels
        assert seen["runtime_gather_output"] is None
        assert seen["config"] is config
        assert seen["compute_language_model_loss"].__self__ is self.gpt_model
        assert seen["scale_logits"].__self__ is self.gpt_model

    @pytest.mark.internal
    def test_build_schedule_plan_threads_output_processor(self):
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        context = {"loss_mask": torch.ones_like(labels)}

        seen = {}

        def output_processor(**kwargs):
            seen.update(kwargs)
            return kwargs["hidden_states"].sum()

        schedule_plan = self.gpt_model.build_schedule_plan(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_processor=output_processor,
            output_processor_context=context,
        )

        assert schedule_plan.state.output_processor is output_processor
        assert schedule_plan.state.output_processor_context is context

        schedule_plan.state.decoder_input = None
        schedule_plan.state.rotary_pos_emb = None
        schedule_plan.state.rotary_pos_cos = None
        schedule_plan.state.rotary_pos_sin = None
        schedule_plan.state.sequence_len_offset = None

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            self.gpt_model.config.hidden_size,
            device="cuda",
            requires_grad=True,
        )
        output = schedule_plan.post_process.forward(hidden_states)

        assert output.item() == pytest.approx(hidden_states.sum().item())
        assert seen["context"] is context
        assert seen["labels"] is labels
        assert seen["output_layer"] is self.gpt_model.output_layer


def test_get_mlp_module_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mlp_module_spec)

    # Define the expected signature
    expected_params = {
        "use_te": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "num_experts": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "fp8": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_te_op_fuser": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "use_te": True,
        "num_experts": None,
        "moe_grouped_gemm": False,
        "fp8": None,
        "use_te_op_fuser": False,
    }

    # Check expected parameters are in function signature
    for param_name, param_kind in expected_params.items():
        assert param_name in sig.parameters, f"Unexpected parameter: {param_name}"
        assert (
            param_kind is sig.parameters[param_name].kind
        ), f"Wrong kind for parameter: {param_name}"

    # Check default values
    sig_defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    for k, v in expected_defaults.items():
        assert (
            k in sig_defaults and v == sig_defaults[k]
        ), f"Default value of {sig_defaults[k]} does not match the expected value of {v} for parameter {k}."


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TEFusedMLP is only supported with TE 1.13+."
)
class TestGPTWithFusedOps:
    """GPT model with Transformer Engine operation-based API"""

    def setup_method(self, method) -> None:
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(use_te_op_fuser=True),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_forward(self) -> None:
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TEFusedMLP is only supported with TE 1.13+."
)
@pytest.mark.parametrize("num_experts", [None, 4])
@pytest.mark.parametrize("gated_linear_unit", [True, False])
def test_gpt_with_te_activation_func(num_experts, gated_linear_unit):
    """Test GPT model with Transformer Engine activation function"""

    # setup
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        use_cpu_initialization=True,
        add_bias_linear=False,
        use_te_activation_func=True,
        bias_activation_fusion=False,
        gated_linear_unit=gated_linear_unit,
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts, use_te_activation_func=True
        ),
        vocab_size=128,
        max_sequence_length=128,
    )

    # test
    sequence_length = gpt_model.max_sequence_length
    micro_batch_size = 2

    gpt_model.cuda()

    data = list(range(sequence_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    attention_mask = torch.ones(
        (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
    ).cuda()

    logits = gpt_model.forward(
        input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
    )

    assert logits.shape[0] == micro_batch_size
    assert logits.shape[1] == sequence_length
    assert logits.shape[2] == gpt_model.vocab_size

    # teardown
    Utils.destroy_model_parallel()


class TestGPTModelWithCustomPG:
    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "tp_size, dp_size, cp_size", [(1, 8, 1), (2, 4, 1)]  # TP 1, DP 8, CP 1  # TP 2, DP 4, CP 1
    )
    def test_gpt_model_with_custom_pg(self, tp_size, dp_size, cp_size):

        # Create HyperCommGrid with dimensions tp, cp, ep, pp, dp (reversed from device mesh order)
        grid = HyperCommGrid([tp_size, cp_size, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])

        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")
        pp_group = grid.create_pg("pp")
        ep_group = grid.create_pg("ep")
        embd_group_ranks = parallel_state.default_embedding_ranks(
            torch.distributed.get_process_group_ranks(pp_group)
        )
        embd_group = torch.distributed.new_group(
            ranks=embd_group_ranks, timeout=timedelta(minutes=30)
        )
        pg_collection = ProcessGroupCollection(
            tp=tp_group, cp=cp_group, pp=pp_group, ep=ep_group, embd=embd_group
        )

        model_parallel_cuda_manual_seed(
            1234, tp_rank=tp_group.rank(), ep_rank=ep_group.rank(), etp_rank=tp_group.rank()
        )
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=1024, num_attention_heads=16, use_cpu_initialization=False
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=512,
            pg_collection=pg_collection,
            post_process=False,
        )

        # Check that model weights are distributed as expected when using TP
        assert (
            self.gpt_model.decoder.layers[0].self_attention.linear_qkv.weight.shape[0]
            == (1024 * 3) / tp_size
        )
        assert self.gpt_model.decoder.layers[0].self_attention.linear_qkv.weight.shape[1] == 1024
        assert self.gpt_model.decoder.layers[0].self_attention.linear_proj.weight.shape[0] == 1024
        assert (
            self.gpt_model.decoder.layers[0].self_attention.linear_proj.weight.shape[1]
            == 1024 / tp_size
        )

        # Check that the logits output shape is correct
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        input_ids = torch.ones(micro_batch_size, sequence_length, dtype=torch.int64, device="cuda")
        position_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.int64, device="cuda"
        )

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        assert logits.shape[0] == sequence_length
        assert logits.shape[1] == micro_batch_size
        assert logits.shape[2] == self.gpt_model.config.hidden_size


class TestGPTWithDynamicInference:
    """Tests GPTModel with dynamic inference."""

    @torch.inference_mode()
    def setup_method(self, method):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=8,
            hidden_size=256,
            num_attention_heads=8,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="tensorwise",
        )

        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=DynamicInferenceContext.TOKEN_ROUNDER,
            parallel_output=True,
        )
        self.gpt_model = Float16Module(self.gpt_model.config, self.gpt_model)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_dynamic_inference_padding_with_fp8(self):
        """
        Tests that logits for padded tokens are zeroed out for fp8 inference.
        """
        self.gpt_model.cuda()
        self.gpt_model.eval()
        config = self.gpt_model.config

        inference_context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=config.params_dtype,
                num_layers=config.num_layers,
                kv_channels=config.hidden_size // config.num_attention_heads,
                num_attention_heads=config.num_attention_heads,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=self.gpt_model.module.max_sequence_length,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
            ),
        )

        # Add a request with 10 tokens. Since 10 is not a multiple of 64,
        # this will create padding up to the padded length of 64.
        active_token_count = 10
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(0, active_token_count, dtype=torch.long, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        inference_context.add_request(request)

        # Prepares the context, including calculating the padded token count.
        inference_context.initialize_attention_state()

        assert inference_context.active_token_count == active_token_count
        assert inference_context.padded_active_token_count == DynamicInferenceContext.TOKEN_ROUNDER

        # Prepare inputs for the forward pass.
        padded_token_count = inference_context.padded_active_token_count
        input_ids, position_ids = inference_context.current_input_and_position_ids()

        # Run the forward pass with inference parameters.
        with InferenceMode.active():
            logits = self.gpt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

        # Verify the output shape.
        assert logits.shape[0] == 1
        assert logits.shape[1] == padded_token_count
        assert logits.shape[2] == self.gpt_model.module.vocab_size

        # Extract the logits corresponding to the padding tokens (from index 10 to 63).
        padding_start_idx = inference_context.active_token_count
        padding_end_idx = inference_context.padded_active_token_count
        padding_logits = logits[0, padding_start_idx:padding_end_idx, :]

        # Assert that all padding logits are zero.
        assert torch.all(padding_logits == 0.0), "Logits for padding tokens are not all zero."


def test_get_transformer_layer_spec_forwards_use_te_activation_func():
    """Test that _get_transformer_layer_spec forwards use_te_activation_func.

    Regression test for https://github.com/NVIDIA/Megatron-LM/issues/2770
    The --use-te-activation-func flag was silently ignored for non-MoE GPT
    models because _get_transformer_layer_spec did not forward the parameter
    to get_gpt_layer_with_transformer_engine_spec.
    """
    mock_config = MagicMock()
    mock_config.use_te_activation_func = True
    mock_config.use_kitchen = False
    mock_config.use_kitchen_attention = False
    mock_config.kitchen_attention_backend = "sdpa"

    mock_args = MagicMock()
    mock_args.num_experts = None
    mock_args.moe_grouped_gemm = False
    mock_args.qk_layernorm = False
    mock_args.multi_latent_attention = False
    mock_args.experimental_attention_variant = None
    mock_args.moe_use_legacy_grouped_gemm = False
    mock_args.qk_l2_norm = False

    with (
        patch('gpt_builders.get_args', return_value=mock_args),
        patch('gpt_builders.get_gpt_layer_with_transformer_engine_spec') as mock_spec_fn,
    ):
        from gpt_builders import _get_transformer_layer_spec

        _get_transformer_layer_spec(use_te=True, config=mock_config)

        mock_spec_fn.assert_called_once()
        _, call_kwargs = mock_spec_fn.call_args
        assert (
            call_kwargs.get('use_te_activation_func') is True
        ), "use_te_activation_func must be forwarded from config"
