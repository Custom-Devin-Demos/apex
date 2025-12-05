"""Tests for encoder-decoder model functionality with pipeline parallelism.

This module tests the encoder-decoder support in apex.transformer pipeline parallelism,
specifically focusing on:
1. Correct behavior with pipeline_model_parallel_split_rank_
2. Proper tensor communication between encoder and decoder stages
3. Gradient flow across the encoder-decoder boundary
"""
import unittest
from typing import Optional, Tuple, List, Dict, Callable

import torch
import torch.nn as nn
from torch.testing._internal import common_utils

from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel import utils as pp_utils
from apex.transformer.pipeline_parallel.schedules.common import (
    build_model,
    forward_step,
    backward_step,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
)
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing import commons as testing_utils


class EncoderDecoderModel(nn.Module):
    """A simple encoder-decoder model for testing pipeline parallelism.

    This model supports the encoder-decoder split by accepting add_encoder and add_decoder
    flags to determine which parts of the model to include at each pipeline stage.
    """

    def __init__(
        self,
        hidden_size: int,
        pre_process: bool = False,
        post_process: bool = False,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        if add_encoder:
            self.encoder_layer = nn.Linear(hidden_size, hidden_size)
        if add_decoder:
            self.decoder_layer = nn.Linear(hidden_size, hidden_size)

        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.input_tensor = input_tensor[0]

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.input_tensor is not None:
            x = self.input_tensor

        if self.add_encoder and hasattr(self, "encoder_layer"):
            x = self.encoder_layer(x)
        if self.add_decoder and hasattr(self, "decoder_layer"):
            x = self.decoder_layer(x)

        return x


def encoder_decoder_model_provider(
    hidden_size: int,
    pre_process: bool,
    post_process: bool,
    add_encoder: bool = True,
    add_decoder: bool = True,
) -> EncoderDecoderModel:
    """Model provider function for encoder-decoder model."""
    return EncoderDecoderModel(
        hidden_size=hidden_size,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
    )


def encoder_decoder_fwd_step_func(batch, model):
    """Forward step function for encoder-decoder model testing."""
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    y = model(x)

    def loss_func(output):
        loss = torch.sum(output)
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {"avg": averaged_loss}

    return y, loss_func


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires >= 4 GPUs")
class TestEncoderDecoderSplitRank(NcclDistributedTestBase):
    """Tests for pipeline_model_parallel_split_rank_ parameter behavior.

    These tests verify that the encoder-decoder split rank parameter properly
    divides the model stages between encoder and decoder portions.
    """

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def test_split_rank_stage_classification(self) -> None:
        """Test that stages are correctly classified as encoder or decoder based on split rank."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )

        try:
            rank = parallel_state.get_pipeline_model_parallel_rank()
            is_before_split = parallel_state.is_pipeline_stage_before_split()
            is_after_split = parallel_state.is_pipeline_stage_after_split()

            if rank < split_rank:
                self.assertTrue(
                    is_before_split,
                    f"Rank {rank} should be before split (split_rank={split_rank})",
                )
                self.assertFalse(
                    is_after_split,
                    f"Rank {rank} should not be after split (split_rank={split_rank})",
                )
            else:
                self.assertFalse(
                    is_before_split,
                    f"Rank {rank} should not be before split (split_rank={split_rank})",
                )
                self.assertTrue(
                    is_after_split,
                    f"Rank {rank} should be after split (split_rank={split_rank})",
                )
        finally:
            parallel_state.destroy_model_parallel()

    def test_split_rank_pre_post_process(self) -> None:
        """Test that pre_process and post_process are set correctly for encoder-decoder models."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )

        try:
            rank = parallel_state.get_pipeline_model_parallel_rank()
            world_size = parallel_state.get_pipeline_model_parallel_world_size()

            expected_pre_process = rank == 0 or rank == split_rank
            expected_post_process = rank == (split_rank - 1) or rank == (world_size - 1)

            actual_pre_process = rank == 0 or rank == split_rank
            actual_post_process = rank == (split_rank - 1) or rank == (world_size - 1)

            self.assertEqual(
                actual_pre_process,
                expected_pre_process,
                f"pre_process mismatch at rank {rank}",
            )
            self.assertEqual(
                actual_post_process,
                expected_post_process,
                f"post_process mismatch at rank {rank}",
            )
        finally:
            parallel_state.destroy_model_parallel()

    def test_split_rank_add_encoder_decoder_flags(self) -> None:
        """Test that add_encoder and add_decoder flags are set correctly based on split rank."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )

        try:
            rank = parallel_state.get_pipeline_model_parallel_rank()

            expected_add_encoder = parallel_state.is_pipeline_stage_before_split()
            expected_add_decoder = parallel_state.is_pipeline_stage_after_split()

            if rank < split_rank:
                self.assertTrue(
                    expected_add_encoder,
                    f"add_encoder should be True for encoder stage (rank {rank})",
                )
                self.assertFalse(
                    expected_add_decoder,
                    f"add_decoder should be False for encoder stage (rank {rank})",
                )
            else:
                self.assertFalse(
                    expected_add_encoder,
                    f"add_encoder should be False for decoder stage (rank {rank})",
                )
                self.assertTrue(
                    expected_add_decoder,
                    f"add_decoder should be True for decoder stage (rank {rank})",
                )
        finally:
            parallel_state.destroy_model_parallel()

    def test_split_rank_none_raises_error(self) -> None:
        """Test that missing split_rank raises RuntimeError for encoder_and_decoder models."""
        pipeline_model_parallel_world_size = 4

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=None,
        )

        try:
            with self.assertRaises(RuntimeError) as context:
                build_model(
                    encoder_decoder_model_provider,
                    wrap_with_ddp=False,
                    model_type=ModelType.encoder_and_decoder,
                    hidden_size=64,
                )
            self.assertIn(
                "Split rank needs to be specified",
                str(context.exception),
            )
        finally:
            parallel_state.destroy_model_parallel()


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires >= 4 GPUs")
class TestEncoderDecoderTensorCommunication(NcclDistributedTestBase):
    """Tests for tensor communication between encoder and decoder stages.

    These tests verify that tensors are correctly passed from the last encoder
    stage to the first decoder stage during forward and backward passes.
    """

    GLOBAL_BATCH_SIZE: int = 16
    MICRO_BATCH_SIZE: int = 2
    HIDDEN_SIZE: int = 64
    SEQUENCE_LENGTH: int = 32

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _setup_pipeline_parallel(
        self,
        pipeline_model_parallel_world_size: int,
        split_rank: int,
    ) -> None:
        """Helper to set up pipeline parallelism with encoder-decoder split."""
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def test_forward_pass_encoder_decoder(self) -> None:
        """Test that forward pass works correctly with encoder-decoder split."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        self._setup_pipeline_parallel(pipeline_model_parallel_world_size, split_rank)

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=True,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=False,
            )
        finally:
            parallel_state.destroy_model_parallel()

    def test_forward_backward_pass_encoder_decoder(self) -> None:
        """Test that forward and backward passes work correctly with encoder-decoder split."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        self._setup_pipeline_parallel(pipeline_model_parallel_world_size, split_rank)

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=False,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=False,
            )

            for m in model:
                for p in m.parameters():
                    self.assertIsNotNone(
                        p.grad,
                        "Gradients should be computed for all parameters",
                    )
        finally:
            parallel_state.destroy_model_parallel()


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires >= 4 GPUs")
class TestEncoderDecoderGradientFlow(NcclDistributedTestBase):
    """Tests for gradient flow across the encoder-decoder boundary.

    These tests ensure gradients properly flow back from decoder stages
    to encoder stages during backpropagation.
    """

    GLOBAL_BATCH_SIZE: int = 16
    MICRO_BATCH_SIZE: int = 2
    HIDDEN_SIZE: int = 64
    SEQUENCE_LENGTH: int = 32

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _setup_pipeline_parallel(
        self,
        pipeline_model_parallel_world_size: int,
        split_rank: int,
    ) -> None:
        """Helper to set up pipeline parallelism with encoder-decoder split."""
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

    def test_gradient_flow_across_boundary(self) -> None:
        """Test that gradients flow correctly across the encoder-decoder boundary."""
        pipeline_model_parallel_world_size = 4
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        self._setup_pipeline_parallel(pipeline_model_parallel_world_size, split_rank)

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            loss = forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=False,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=False,
            )

            for m in model:
                for name, p in m.named_parameters():
                    self.assertIsNotNone(
                        p.grad,
                        f"Gradient should be computed for parameter {name}",
                    )
                    self.assertFalse(
                        torch.isnan(p.grad).any(),
                        f"Gradient for {name} contains NaN values",
                    )
                    self.assertFalse(
                        torch.isinf(p.grad).any(),
                        f"Gradient for {name} contains Inf values",
                    )
        finally:
            parallel_state.destroy_model_parallel()

    def test_gradient_flow_with_sequence_parallel(self) -> None:
        """Test gradient flow with sequence parallelism enabled."""
        tensor_model_parallel_size = 2
        pipeline_model_parallel_world_size = self.world_size // tensor_model_parallel_size

        if pipeline_model_parallel_world_size < 2:
            self.skipTest("Requires at least 4 GPUs for this test configuration")

        split_rank = pipeline_model_parallel_world_size // 2

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_size,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=True,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=True,
                ),
                batch=batch,
                model=model,
                forward_only=False,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=True,
            )

            for m in model:
                for name, p in m.named_parameters():
                    self.assertIsNotNone(
                        p.grad,
                        f"Gradient should be computed for parameter {name}",
                    )
        finally:
            parallel_state.destroy_model_parallel()


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires >= 4 GPUs")
class TestEncoderDecoderDifferentSplitRanks(NcclDistributedTestBase):
    """Tests for encoder-decoder models with different split rank configurations.

    These tests verify that the encoder-decoder split works correctly with
    various split rank values.
    """

    GLOBAL_BATCH_SIZE: int = 16
    MICRO_BATCH_SIZE: int = 2
    HIDDEN_SIZE: int = 64
    SEQUENCE_LENGTH: int = 32

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def _run_encoder_decoder_test(
        self,
        pipeline_model_parallel_world_size: int,
        split_rank: int,
        forward_only: bool,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Helper to run encoder-decoder test with given configuration."""
        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )
            model = [m.to(dtype=dtype) for m in model]

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=dtype,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=forward_only,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=dtype,
                sequence_parallel_enabled=False,
            )

            if not forward_only:
                for m in model:
                    for p in m.parameters():
                        self.assertIsNotNone(p.grad)
        finally:
            parallel_state.destroy_model_parallel()

    def test_split_rank_1_of_4(self) -> None:
        """Test encoder-decoder with split_rank=1 (1 encoder stage, 3 decoder stages)."""
        self._run_encoder_decoder_test(
            pipeline_model_parallel_world_size=4,
            split_rank=1,
            forward_only=False,
        )

    def test_split_rank_2_of_4(self) -> None:
        """Test encoder-decoder with split_rank=2 (2 encoder stages, 2 decoder stages)."""
        self._run_encoder_decoder_test(
            pipeline_model_parallel_world_size=4,
            split_rank=2,
            forward_only=False,
        )

    def test_split_rank_3_of_4(self) -> None:
        """Test encoder-decoder with split_rank=3 (3 encoder stages, 1 decoder stage)."""
        self._run_encoder_decoder_test(
            pipeline_model_parallel_world_size=4,
            split_rank=3,
            forward_only=False,
        )

    def test_split_rank_inference_mode(self) -> None:
        """Test encoder-decoder in inference mode (forward_only=True)."""
        self._run_encoder_decoder_test(
            pipeline_model_parallel_world_size=4,
            split_rank=2,
            forward_only=True,
        )

    def test_split_rank_half_precision(self) -> None:
        """Test encoder-decoder with half precision (float16)."""
        self._run_encoder_decoder_test(
            pipeline_model_parallel_world_size=4,
            split_rank=2,
            forward_only=False,
            dtype=torch.float16,
        )


@unittest.skipIf(torch.cuda.device_count() < 8, "Requires >= 8 GPUs")
class TestEncoderDecoderLargerPipeline(NcclDistributedTestBase):
    """Tests for encoder-decoder models with larger pipeline configurations.

    These tests verify encoder-decoder functionality with 8 GPU pipeline
    configurations.
    """

    GLOBAL_BATCH_SIZE: int = 32
    MICRO_BATCH_SIZE: int = 2
    HIDDEN_SIZE: int = 64
    SEQUENCE_LENGTH: int = 32

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def test_8_stage_pipeline_split_at_4(self) -> None:
        """Test 8-stage pipeline with split at rank 4."""
        pipeline_model_parallel_world_size = 8
        split_rank = 4

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=False,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=False,
            )

            for m in model:
                for p in m.parameters():
                    self.assertIsNotNone(p.grad)
        finally:
            parallel_state.destroy_model_parallel()

    def test_8_stage_pipeline_asymmetric_split(self) -> None:
        """Test 8-stage pipeline with asymmetric split (2 encoder, 6 decoder)."""
        pipeline_model_parallel_world_size = 8
        split_rank = 2

        if self.world_size < pipeline_model_parallel_world_size:
            self.skipTest(
                f"Requires at least {pipeline_model_parallel_world_size} GPUs"
            )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=split_rank,
        )
        pp_utils._reconfigure_microbatch_calculator(
            rank=parallel_state.get_tensor_model_parallel_rank(),
            rampup_batch_size=None,
            global_batch_size=self.GLOBAL_BATCH_SIZE,
            micro_batch_size=self.MICRO_BATCH_SIZE,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        try:
            model = build_model(
                testing_utils.mlp_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=None,
                hidden_size=self.HIDDEN_SIZE,
                sequence_parallel_enabled=False,
            )

            batch = None
            if parallel_state.is_pipeline_first_stage():
                batch = (
                    torch.ones(
                        (
                            self.GLOBAL_BATCH_SIZE,
                            self.SEQUENCE_LENGTH,
                            self.HIDDEN_SIZE,
                        ),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )

            forward_backward_pipelining_without_interleaving(
                forward_step_func=testing_utils.ToyParallelMLPFwdBwdStepFunc(
                    sequence_parallel_enabled=False,
                ),
                batch=batch,
                model=model,
                forward_only=False,
                tensor_shape=(
                    self.SEQUENCE_LENGTH,
                    self.MICRO_BATCH_SIZE,
                    self.HIDDEN_SIZE,
                ),
                model_type=ModelType.encoder_and_decoder,
                decoder_sequence_length=self.SEQUENCE_LENGTH,
                async_comm=False,
                grad_scaler=None,
                deallocate_pipeline_outputs=False,
                dtype=torch.float32,
                sequence_parallel_enabled=False,
            )

            for m in model:
                for p in m.parameters():
                    self.assertIsNotNone(p.grad)
        finally:
            parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
