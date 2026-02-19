import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

'''
## Structure of the Kernel

1. **Prologue**: The phase before the first MMA instructions. It usually defines, fetches, allocates, partitions or calculates necessary components (listed below). 
   What else, load multiple stages of data ahead of the first MMA to help hide GMEM latency.
   - Indexing
     * `block_idx` (bidx, bidy): Block index in the grid
     * `mma_coord_mnk`: The location of which block the current MMA unit will calculate (see details in figure 1)
     * `thread_idx` (tidx): Thread index within a block (0 to threads_per_cta - 1). We need this to slice the partition of tensor memory for each thread in a block 
     (see details in [PTX Document 9.7.16.2.3.1 Memory Layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-layout))
     * `warp_idx`: As TMA & tcgen05.mma only needs one thread to issue, some code only needs to execute by warp 0
   - Allocation
     * `smem` (storage, sA, sB): Allocate necessary smem usage for pipelines, A/B smem tensors as input of tcgen05.mma
     * `tmem`: Allocate necessary tmem usage for Acc
   - Pipeline (see more details in async_pipeline.ipynb)
     * `PipelineTmaUmma`: Tma & tcgen05.mma units are async. PipelineTmaUmma helps notify: 
     1. tcgen05.mma when TMA fills A/B buffers to full; 
     2. TMA when tcgen05.mma consumes A/B buffer to empty
     * `PipelineUmmaAsync`: It helps threads when tcgen05.mma finish the accumulation and Acc is ready
     * `Barrier initialization`: barrier initialization work is done inside the pipeline create functions
   - Partition
     * `local_tile`: Get the block of A/B/C GMEM tensors for current MMA unit acoording to `mma_coord_mnk`.
     * `TMA`: Get the tensor view from each TMA instruction
     * `MMA`: Get the tensor view from each tcgen05.mma instruction
   - TMA descriptor prefetch
     * `cpasync.prefetch_descriptor`: helps shorten the latency of access tma descrptor, i.e. tma_atom_a, tma_atom_b

2. **Mainloop**: The phase that carries out the main computation of GEMM. It's usually organized as a loop to iterate blocks in K dim for accumulation. 
   The loop body contains:
    - `Data prefetch` with a fixed stride (ab_stage - 1) ahead of current K block
    - `MMA computation` for current K block

3. **Epilogue**: The phase after the MMA instructions finish the accumulation. It usually contains:
    - `Partition`: Get the tensor views from epi tiler (acc subtile) & each tcgen05.ld instruction
    - `Acc fetch`: Load data from tensor memory to register
    - `Fusion & datatype conversion`: Fuse some operations on C (optional); Datatype conversion if output type is different from acc type
    - `Relinquish tmem alloc permit`: Give permit for following launched kernels
    - `Storing`: TMA or st.global to store out
    - `TMEM deallocation`: Deallocate tmem for Acc buffer
    
    Usually, we subtile the acc buffer to save resources of registers & smem (if using TMA to store C). 
    For our mma_tiler (128, 256), each thread needs 256 registers if no subtiling. 
    Besides, better instruction-level parallelism for interleavely issuing tcgen05.ld, data conversion & st.global.

        python
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(io_dtype))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
'''

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    #
    # 1. Prepare args
    #

    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA and make fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 32x128 bits
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. Main loop
    #
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # Wait for a empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            # Execute one K-block worth of MMA instructions
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal that the A/B buffers have been consumed and are ready for the next load
            ab_full.release()

        # Signal that the accumulator is fully computed
        acc_empty.commit()

    #
    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    kernel: cutlass.Constexpr,
):
    # Tensors and Tiler Setup
    io_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    mma_inst_shape_mnk = (128, 256, 16)
    mma_tiler_mnk = (128, 256, 64)
    threads_per_cta = 128

    # Pipeline stage configuration
    ab_stages = 4
    acc_stage = 1

    m, n, k = 8192, 8192, 8192

    # Make K-major tensors (torch tensors are row-major)
    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_tensor = (
        from_dlpack(a)
        .mark_layout_dynamic(leading_dim=1)
    )
    b_tensor = (
        from_dlpack(b)
        .mark_layout_dynamic(leading_dim=1)
    )
    c_tensor = (
        from_dlpack(c)
        .mark_layout_dynamic(leading_dim=1)
    )

    # Construct tiled MMA
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )

    #### --- MMA Atom Construction --- ####
    # MMA involves 4 tensors: D = A @ B + C
    # The MMA Atom captures the Operation (Number of Regs for each tensor) 
    # ... and the Traits:
    # ...... 1.) Types per tensor
    # ...... 2.) Thread Layout [MMA Local thread_idx -> thread_hierarchy_idx mapping]
    # ...... 3.) TV Layouts per tensor
    # ...... 4.) MMA Shape (M, N, K)
   
    tiled_mma = cute.make_tiled_mma(op)

    #### --- Construct SMEM layouts for A and B --- ###
    # mma_tiler_mnk = (128, 256, 64) -> CTA Tiler
    # SMEM Layout of A -> (m, k, stages) -> (128, 64, 4)
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # Construct TMA load atoms
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )

host_function(a_tensor, b_tensor, c_tensor, None)
