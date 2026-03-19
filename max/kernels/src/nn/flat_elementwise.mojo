# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Optimized flat-index elementwise GPU kernel for contiguous buffers.

Avoids the expensive multi-dimensional index decomposition (integer
divisions) used by the generic `algorithm.functional.elementwise` by
iterating with flat indices over contiguous memory.
"""

from math import ceildiv, clamp
from sys import simd_width_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.primitives.grid_controls import pdl_launch_attributes
from gpu.host import DeviceContext
from gpu.host.info import B200

from utils.static_tuple import StaticTuple


fn flat_elementwise_add[
    type: DType, simd_width: Int, block_size_override: Int = 0
](
    output_ptr: UnsafePointer[Scalar[type], MutAnyOrigin],
    input0_ptr: UnsafePointer[Scalar[type], ImmutAnyOrigin],
    input1_ptr: UnsafePointer[Scalar[type], ImmutAnyOrigin],
    num_elements: Int,
    ctx: DeviceContext,
) raises:
    """Elementwise add using flat indexing for contiguous buffers.

    Eliminates the per-element integer division overhead of generic
    elementwise by using direct pointer arithmetic with flat indices.

    Parameters:
        type: The element data type.
        simd_width: Number of elements processed per vector operation.

    Args:
        output_ptr: Device pointer to the output buffer.
        input0_ptr: Device pointer to the first input buffer.
        input1_ptr: Device pointer to the second input buffer.
        num_elements: Total number of elements.
        ctx: The GPU device context.
    """
    comptime hw_info = ctx.default_device_info
    comptime registers_per_thread = 255
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count = UInt(hw_info.sm_count)
    comptime threads_per_multiprocessor = UInt(
        hw_info.threads_per_multiprocessor
    )
    comptime num_waves = 32

    var length = UInt(num_elements)
    var num_packed = length // UInt(simd_width)
    var tail_len = length % UInt(simd_width)
    var packed_region = length - tail_len

    if length == 0:
        return

    comptime block_size_unrounded = registers_per_block // registers_per_thread
    comptime block_size_auto = 128 if ctx.default_device_info == B200 else block_size_unrounded - (
        block_size_unrounded % 2
    )
    comptime block_size = block_size_override if block_size_override > 0 else block_size_auto

    var num_blocks = clamp(
        ceildiv(num_packed, UInt(block_size)),
        1,
        sm_count * threads_per_multiprocessor // UInt(block_size) * num_waves,
    )

    @__copy_capture(num_packed, tail_len, packed_region)
    @parameter
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(block_size)
        )
    )
    fn _flat_add_kernel[*, block_size: UInt]():
        var tid = thread_idx.x + block_size * block_idx.x

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_BEGINNING:
            launch_dependent_grids()

        comptime if PDLLevel() > PDLLevel.OFF:
            wait_on_dependent_grids()

        # Main loop: process packed (vectorized) elements with flat indexing.
        # No integer divisions needed -- just multiply by simd_width.
        for packed_idx in range(tid, num_packed, block_size * grid_dim.x):
            var flat_off = Int(packed_idx * UInt(simd_width))
            var v0 = input0_ptr.load[width=simd_width](flat_off)
            var v1 = input1_ptr.load[width=simd_width](flat_off)
            output_ptr.store[width=simd_width](flat_off, v0 + v1)

        # Tail: scalar elements that don't fill a full SIMD vector.
        if tid < tail_len:
            var off = Int(packed_region + tid)
            output_ptr[off] = input0_ptr[off] + input1_ptr[off]

        comptime if PDLLevel() == PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

    comptime kernel = _flat_add_kernel[block_size=UInt(block_size)]
    ctx.enqueue_function[kernel, kernel](
        grid_dim=Int(num_blocks),
        block_dim=block_size,
        attributes=pdl_launch_attributes(),
    )
