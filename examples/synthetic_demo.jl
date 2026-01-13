using CUDA
using LinearAlgebra
using DiagonalODEOpt

CUDA.allowscalar(false)

# --- synthetic problem ---
n = 256
S = 128

# A0: random sparse-ish stable-ish operator
A0 = CUDA.randn(Float64, n, n) .* 0.01
@views A0[diagind(A0)] .= 0.0

# forcing vectors (constant f)
F = CUDA.randn(Float64, n, S)

# labels: toy "class = argmax state index" style
labels = collect(1:S) .% n .+ 1   # just an example mapping

# θ init
θ = CUDA.zeros(Float64, n)

# choose diagonal parametrization
param = SoftplusParam()

# train
θ, hist = optimize_diagonal_adam!(
    A0, θ, F, labels, param;
    T=2.0, Q=32, epochs=10,
    lr=1e-3, k=30, tol=1e-10, β_soft=10.0,
    batchsize=64,
    weight_decay=0.0,
    clamp_theta=(-50.0, 50.0),
    callback = (ep, loss, θ) -> println("ep=$ep loss=$(round(loss, sigdigits=6))")
)

println("done; final loss = ", hist[end])
