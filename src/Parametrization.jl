module Parametrization

export AbstractDiagParametrization,
       ExpParam,
       SoftplusParam,
       diag_from_theta,
       dtheta_from_ddiag,
       softplus

# ----------------------------
# Stable softplus
# ----------------------------
@inline function softplus(x::T) where {T}
    return max(x, zero(T)) + log1p(exp(-abs(x)))
end

# ----------------------------
# Parametrization interface
# ----------------------------

abstract type AbstractDiagParametrization end

"""
    ExpParam()

d(θ) = -exp(θ)

Strong monotone damping, very stiff but simple derivative.
"""
struct ExpParam <: AbstractDiagParametrization end

"""
    SoftplusParam()

d(θ) = -softplus(θ)

Smoother near zero, weaker damping for large negative θ.
"""
struct SoftplusParam <: AbstractDiagParametrization end


# --- mapping θ -> diagonal ---

diag_from_theta(::ExpParam, θ) = -exp.(θ)
diag_from_theta(::SoftplusParam, θ) = -softplus.(θ)

# --- chain rule: ∂d / ∂θ ---

dtheta_from_ddiag(::ExpParam, θ) = -exp.(θ)

# derivative of softplus is sigmoid
@inline σ(x) = inv(one(x) + exp(-x))
dtheta_from_ddiag(::SoftplusParam, θ) = -σ.(θ)

end # module
