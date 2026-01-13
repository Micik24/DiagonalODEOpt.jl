module AdamOptimizer

export AdamState, adam_init, adam_step!

using CUDA

mutable struct AdamState{T}
    m::CuArray{T,1}
    v::CuArray{T,1}
    t::Int
    β1t::T
    β2t::T
end

function adam_init(θ::CuArray{T,1};
                   β1::T=T(0.9),
                   β2::T=T(0.999)) where {T}
    return AdamState{T}(
        CUDA.zeros(T, length(θ)),
        CUDA.zeros(T, length(θ)),
        0,
        one(T),   # β1^t
        one(T)    # β2^t
    )
end

"""
    adam_step!(θ, g, st; lr, β1, β2, ϵ, weight_decay)

GPU-safe Adam update. `g` may be modified in-place if weight_decay != 0.
"""
function adam_step!(
    θ::CuArray{T,1},
    g::CuArray{T,1},
    st::AdamState{T};
    lr::T,
    β1::T = T(0.9),
    β2::T = T(0.999),
    ϵ::T  = T(1e-8),
    weight_decay::T = T(0)
) where {T}

    st.t += 1
    st.β1t *= β1
    st.β2t *= β2

    if weight_decay != 0
        @. g = g + weight_decay * θ
    end

    @. st.m = β1 * st.m + (1 - β1) * g
    @. st.v = β2 * st.v + (1 - β2) * (g * g)

    mhat = st.m ./ (1 - st.β1t)
    vhat = st.v ./ (1 - st.β2t)

    @. θ = θ - lr * mhat / (sqrt(vhat) + ϵ)
    return θ
end

end # module
