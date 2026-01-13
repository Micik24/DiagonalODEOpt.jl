module AdjointStep

export step_grad_theta!

using CUDA
using LinearAlgebra

using ..Parametrization
using ..SchurCPU
using ..ReconstructionGPU

using Batched_Krylov_GPU: batched_arnoldi_cgs2_gpu


# ------------------------------------------------------------
# Stable softmax + CE on CPU
# ------------------------------------------------------------

function softmax_ce_and_lambdaT_cpu(
    xT::CuArray{Float64,2},
    labels::Vector{Int},
    β::Float64;
    active::Union{Nothing,AbstractVector{Bool}} = nothing
)
    x = Array(xT)
    n, S = size(x)

    if active === nothing
        active = trues(S)
    end

    lambda = zeros(Float64, n, S)
    loss_sum = 0.0
    active_cnt = 0

    @inbounds for s in 1:S
        if !active[s]; continue; end

        ys = labels[s]

        m = -Inf
        for i in 1:n
            v = β * x[i, s]
            if v > m; m = v; end
        end

        denom = 0.0
        for i in 1:n
            pi = exp(β * x[i, s] - m)
            denom += pi
            lambda[i, s] = pi
        end

        invden = 1.0 / denom
        for i in 1:n
            lambda[i, s] = β * (lambda[i, s] * invden)
        end

        p_true = (lambda[ys, s] / β)
        loss_sum -= log(max(p_true, eps(Float64)))

        lambda[ys, s] -= β
        active_cnt += 1
    end

    loss = active_cnt == 0 ? 0.0 : (loss_sum / active_cnt)
    return loss, CuArray(lambda)
end


# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------

"""
    step_grad_theta!(A0, theta, rhs, labels, param; ...)

Compute loss and gradient w.r.t. θ for a batched linear ODE using
GPU Arnoldi + CPU Schur exponential integrators.
"""
function step_grad_theta!(
    A0::CuArray{Float64,2},
    theta::CuArray{Float64,1},
    rhs::CuArray{Float64,2},
    labels::Vector{Int},
    param::AbstractDiagParametrization;
    T::Float64,
    times::Vector{Float64},
    weights_gpu::CuArray{Float64,1},
    k::Int = 30,
    tol::Float64 = 1e-14,
    β_soft::Float64 = 10.0
)

    n, S = size(rhs)
    Q = length(times)

    # ------------------------------------------------------------
    # Diagonal parametrization
    # ------------------------------------------------------------
    diagA = diag_from_theta(param, theta)

    # ------------------------------------------------------------
    # Forward Arnoldi
    # ------------------------------------------------------------
    Vall_f, Hall_f, beta_f, active_f =
        batched_arnoldi_cgs2_gpu(A0, diagA, rhs; kmax=k, tol=tol)

    active_f_cpu = Array(active_f)
    Hall_f_cpu = Array(@view Hall_f[1:k, 1:k, :])

    # ------------------------------------------------------------
    # Reconstruct x(T)
    # ------------------------------------------------------------
    schur_f_T = build_schur_cache(Hall_f_cpu)

    coeffT_cpu = Array{Float64,3}(undef, k, 1, S)
    tmpk = zeros(Float64, k)
    for s in 1:S
        phi1_t_e1_schur!(tmpk, schur_f_T[s], T)
        @views coeffT_cpu[:, 1, s] .= tmpk
    end

    coeffT = CuArray(coeffT_cpu)
    xT = CUDA.zeros(Float64, n, S)

    reconstruct_times!(reshape(xT, n, 1, S),
                       Vall_f, coeffT, beta_f, active_f;
                       k=k, kmax=k)

    # ------------------------------------------------------------
    # Preliminary lambda(T)
    # ------------------------------------------------------------
    _, lambdaT_pre =
        softmax_ce_and_lambdaT_cpu(xT, labels, β_soft; active=active_f_cpu)

    # ------------------------------------------------------------
    # Adjoint Arnoldi #1 (mask refinement)
    # ------------------------------------------------------------
    diagAdj = -diagA

    _, _, _, active_l_pre =
        batched_arnoldi_cgs2_gpu(A0, diagAdj, lambdaT_pre;
                                 kmax=k, tol=tol,
                                 transA0=true, α=-1.0)

    active_batch_cpu = active_f_cpu .& Array(active_l_pre)

    # ------------------------------------------------------------
    # Final lambda(T) on unified mask
    # ------------------------------------------------------------
    loss, lambdaT =
        softmax_ce_and_lambdaT_cpu(xT, labels, β_soft;
                                   active=active_batch_cpu)

    # ------------------------------------------------------------
    # Adjoint Arnoldi #2
    # ------------------------------------------------------------
    Vall_l, Hall_l, beta_l, active_l =
        batched_arnoldi_cgs2_gpu(A0, diagAdj, lambdaT;
                                 kmax=k, tol=tol,
                                 transA0=true, α=-1.0)

    active_batch_cpu .&= Array(active_l)
    loss, lambdaT =
        softmax_ce_and_lambdaT_cpu(xT, labels, β_soft;
                                   active=active_batch_cpu)

    active_batch = CuArray(active_batch_cpu)

    # ------------------------------------------------------------
    # Reconstruct Lambda(t)
    # ------------------------------------------------------------
    Hall_l_cpu = Array(@view Hall_l[1:k, 1:k, :])
    deltas = T .- times

    schur_l = build_schur_cache(Hall_l_cpu; ts=deltas)
    coeffs_exp = CuArray(build_exp_coeffs_schur(schur_l, deltas))

    Lambda = CUDA.zeros(Float64, n, Q, S)
    reconstruct_times!(Lambda, Vall_l, coeffs_exp,
                       beta_l, active_batch; k=k, kmax=k)

    # ------------------------------------------------------------
    # Reconstruct X(t)
    # ------------------------------------------------------------
    schur_f = build_schur_cache(Hall_f_cpu; ts=times)
    coeffs_phi1 = CuArray(build_phi1_coeffs_schur(schur_f, times))

    X = CUDA.zeros(Float64, n, Q, S)
    reconstruct_times!(X, Vall_f, coeffs_phi1,
                       beta_f, active_batch; k=k, kmax=k)

    # ------------------------------------------------------------
    # Gradient assembly
    # ------------------------------------------------------------
    Z  = Lambda .* X
    ZS = sum(Z; dims=3)

    w = reshape(weights_gpu, 1, Q, 1)
    grad_d = vec(sum(ZS .* w; dims=2))

    # Chain rule through parametrization
    grad_theta = grad_d .* dtheta_from_ddiag(param, theta)

    return loss, grad_theta
end

end # module
