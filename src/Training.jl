module Training

export optimize_diagonal_adam!

using CUDA
using LinearAlgebra

using ..Parametrization: AbstractDiagParametrization
using ..AdjointStep: step_grad_theta!
using ..AdamOptimizer: AdamState, adam_init, adam_step!

CUDA.allowscalar(false)

"""
    optimize_diagonal_adam!(
        A0, θ, F, labels, param;
        T, Q, epochs, lr,
        k, tol, β_soft, batchsize,
        weight_decay, clamp_theta,
        callback
    ) -> (θ, history)

High-level training loop for diagonal parameters θ using Adam.
- Forward/adjoint handled by `step_grad_theta!`.
- `labels` are global indices (no relabeling inside batches).
- `callback(ep, loss, θ)` optional.
"""
function optimize_diagonal_adam!(
    A0::CuArray{Float64,2},
    θ::CuArray{Float64,1},
    F::CuArray{Float64,2},
    labels::Vector{Int},
    param::AbstractDiagParametrization;
    T::Float64,
    Q::Int = 32,
    epochs::Int = 20,
    lr::Float64 = 1e-3,
    k::Int = 30,
    tol::Float64 = 1e-14,
    β_soft::Float64 = 10.0,
    batchsize::Int = 0,                 # 0 => full batch
    weight_decay::Float64 = 0.0,
    clamp_theta::Union{Nothing,Tuple{Float64,Float64}} = nothing,
    callback::Union{Nothing,Function} = nothing
)
    n, S = size(F)
    @assert length(labels) == S "labels length must match number of columns in F"

    # time grid + weights on GPU
    times = collect(range(0.0, T; length=Q+1))
    h = T / Q
    weights = fill(h, Q+1)
    weights[1] *= 0.5
    weights[end] *= 0.5
    weights_gpu = CuArray(weights)

    st = adam_init(θ)

    history = Float64[]

    # batching indices
    all_idxs = collect(1:S)
    batches = if batchsize <= 0 || batchsize >= S
        [all_idxs]
    else
        [all_idxs[i:min(i+batchsize-1, S)] for i in 1:batchsize:S]
    end

    for ep in 1:epochs
        ep_loss = 0.0

        for idxs in batches
            Fb = @view F[:, idxs]
            lb = labels[idxs]

            loss, gθ = step_grad_theta!(
                A0, θ, CuArray(Fb), lb, param;
                T=T, times=times, weights_gpu=weights_gpu,
                k=k, tol=tol, β_soft=β_soft
            )

            adam_step!(θ, gθ, st; lr=lr, weight_decay=weight_decay)

            if clamp_theta !== nothing
                lo, hi = clamp_theta
                @. θ = clamp(θ, lo, hi)
            end

            ep_loss += loss
        end

        ep_loss /= length(batches)
        push!(history, ep_loss)

        if callback !== nothing
            callback(ep, ep_loss, θ)
        end
    end

    return θ, history
end

end # module
