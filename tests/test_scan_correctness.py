# tests/test_scan_correctness.py
"""Proves selective_scan_chunked ≈ selective_scan_naive."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.mamba_scan import selective_scan_naive, selective_scan_chunked  # noqa: E402


def _make_random_inputs(Bsz, L, d_inner, N, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(Bsz, L, d_inner, generator=g)
    # delta must be positive (it's a softplus output in the real model)
    delta = torch.nn.functional.softplus(torch.randn(Bsz, L, d_inner, generator=g) * 0.5 - 1.0)
    # A must be negative real (S4D-real init: -exp(A_log))
    A = -torch.exp(torch.randn(d_inner, N, generator=g) * 0.3)
    B_sel = torch.randn(Bsz, L, N, generator=g) * 0.5
    C_sel = torch.randn(Bsz, L, N, generator=g) * 0.5
    D = torch.randn(d_inner, generator=g) * 0.1 + 1.0
    return (t.to(device) for t in (x, delta, A, B_sel, C_sel, D))


def _check(Bsz, L, d_inner, N, chunk_size, atol=1e-4, rtol=1e-4):
    x, delta, A, B_sel, C_sel, D = _make_random_inputs(Bsz, L, d_inner, N, seed=Bsz + L + d_inner + N + chunk_size)
    y_naive, h_naive = selective_scan_naive(x, delta, A, B_sel, C_sel, D)
    y_chunk, h_chunk = selective_scan_chunked(x, delta, A, B_sel, C_sel, D, chunk_size=chunk_size)

    assert y_naive.shape == y_chunk.shape == (Bsz, L, d_inner)
    max_abs_err = (y_naive - y_chunk).abs().max().item()
    assert torch.allclose(y_naive, y_chunk, atol=atol, rtol=rtol), (
        f"y mismatch: max_abs_err={max_abs_err} (B={Bsz},L={L},d={d_inner},N={N},chunk={chunk_size})"
    )
    assert torch.allclose(h_naive, h_chunk, atol=atol, rtol=rtol), (
        f"final state mismatch (B={Bsz},L={L},d={d_inner},N={N},chunk={chunk_size})"
    )


def test_small_exact_division():
    _check(Bsz=2, L=128, d_inner=8, N=4, chunk_size=32)


def test_chunk_not_dividing_length():
    _check(Bsz=2, L=100, d_inner=8, N=4, chunk_size=16)  # 100 % 16 != 0


def test_chunk_size_one_equals_naive():
    _check(Bsz=1, L=17, d_inner=4, N=4, chunk_size=1)


def test_larger_shapes():
    _check(Bsz=3, L=257, d_inner=32, N=16, chunk_size=64)


def test_single_step():
    _check(Bsz=2, L=1, d_inner=8, N=4, chunk_size=64)


def test_carry_state_h0():
    Bsz, L, d_inner, N = 2, 50, 8, 4
    x, delta, A, B_sel, C_sel, D = _make_random_inputs(Bsz, L, d_inner, N, seed=42)
    h0 = torch.randn(Bsz, d_inner, N) * 0.1
    y_naive, h_naive = selective_scan_naive(x, delta, A, B_sel, C_sel, D, h0=h0)
    y_chunk, h_chunk = selective_scan_chunked(x, delta, A, B_sel, C_sel, D, chunk_size=13, h0=h0)
    assert torch.allclose(y_naive, y_chunk, atol=1e-4, rtol=1e-4)
    assert torch.allclose(h_naive, h_chunk, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_small_exact_division()
    test_chunk_not_dividing_length()
    test_chunk_size_one_equals_naive()
    test_larger_shapes()
    test_single_step()
    test_carry_state_h0()
    print("All scan correctness tests passed.")