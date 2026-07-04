"""Device + kernel sanity for the two GPU boxes (DEVPLAN P0.3).

Prints device name and compute capability, runs a 4096x4096 fp16 matmul and an
``F.scaled_dot_product_attention`` call, asserts neither produced NaN, and
reports which SDPA backend was chosen. Run on both machines and paste both
outputs into the README.

Volta note (Appendix C): on the V100 node this must pick a non-Flash SDPA
backend (mem-efficient or math) — FlashAttention never supported sm_70.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def main() -> int:
    print(f"torch {torch.__version__} | cuda build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    capability = f"sm_{props.major}{props.minor}"
    print(f"device: {props.name} | capability {capability} | {props.total_memory / 2**30:.1f} GiB")

    # 4096x4096 fp16 matmul.
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    assert torch.isfinite(c.float()).all(), "fp16 matmul produced non-finite values"
    print(f"fp16 matmul 4096x4096: ok (mean abs {c.float().abs().mean().item():.2f})")

    # Scaled dot-product attention: report the backend actually usable here,
    # then run one call under autocast and check for NaN.
    q = torch.randn(4, 8, 1024, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    backends = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    print(f"sdpa backends enabled: {backends}")

    chosen = None
    for name, ctx_backend in (
        ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
        ("math", torch.nn.attention.SDPBackend.MATH),
    ):
        try:
            with torch.nn.attention.sdpa_kernel(ctx_backend):
                out = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            chosen = name
            break
        except RuntimeError:
            continue
    if chosen is None:
        print("FAIL: no SDPA backend could run")
        return 1
    assert torch.isfinite(out.float()).all(), "SDPA produced non-finite values"
    print(f"sdpa: ok | first usable backend: {chosen}")

    print("gpu_sanity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
