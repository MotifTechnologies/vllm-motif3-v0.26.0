/*
 * Always-compiled entry point for the motif3 grouped-PolyNorm NVFP4 quant op.
 *
 * The kernel itself (grouped_poly_norm_nvfp4_quant_kernel.cu) is only compiled
 * when FP4_ARCHS is non-empty, so its symbol is absent on builds without an
 * NVFP4-capable arch. This file is always compiled and always registers the op,
 * failing with a clear message at call time instead of at link time — the same
 * split nvfp4_quant_entry.cu uses for scaled_fp4_quant.
 *
 * LOCAL PATCH (2026-08-18): upstream ships this op for SM10x only -- the kernel
 * was listed solely in FP4_SM100_SRCS and the runtime check below read
 * `sm >= 100 && sm < 120`, which rejects SM120 even when a kernel is present.
 * On RTX PRO 6000 Blackwell (SM120) that made every worker abort at startup
 * with "No compiled grouped_poly_norm_nvfp4_quant kernel". The kernel has no
 * arch-specific code (no tcgen05/wgmma/__CUDA_ARCH__ branches) and its only fp4
 * dependency, quantization/fp4/nvfp4_utils.cuh, is arch-neutral and already
 * compiled into the SM12x group for other kernels, so it is now built for SM12x
 * as well and accepted here. The SM10x path is unchanged.
 */

#include <torch/csrc/stable/tensor.h>

#include "torch_utils.h"

#include "cutlass_extensions/common.hpp"

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  #define VLLM_HAS_GROUPED_POLY_NORM_NVFP4 1
#endif

#if defined(VLLM_HAS_GROUPED_POLY_NORM_NVFP4)
void grouped_poly_norm_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input, torch::stable::Tensor const& mul,
    torch::stable::Tensor const& weight, torch::stable::Tensor const& bias,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& blockscale_offsets,
    torch::stable::Tensor const& input_global_scale, double eps,
    double hidden_clamp, double polynorm_output_scale);
#endif

void grouped_poly_norm_nvfp4_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input, torch::stable::Tensor const& mul,
    torch::stable::Tensor const& weight, torch::stable::Tensor const& bias,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& blockscale_offsets,
    torch::stable::Tensor const& input_global_scale, double eps,
    double hidden_clamp, double polynorm_output_scale) {
#if defined(VLLM_HAS_GROUPED_POLY_NORM_NVFP4)
  const int32_t sm = get_sm_version_num();
  bool supported = false;
  #if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  supported = supported || (sm >= 100 && sm < 120);
  #endif
  #if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  supported = supported || (sm >= 120 && sm < 130);
  #endif
  STD_TORCH_CHECK(supported,
                  "No compiled grouped_poly_norm_nvfp4_quant kernel for SM ",
                  sm, ". Recompile with the appropriate CUDA arch.");
  return grouped_poly_norm_nvfp4_quant_sm1xxa(
      output, output_scale, input, mul, weight, bias, expert_offsets,
      blockscale_offsets, input_global_scale, eps, hidden_clamp,
      polynorm_output_scale);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled grouped_poly_norm_nvfp4_quant kernel");
}
