module SchurCPU

export SchurCache,
       build_schur_cache,
       exp_e1_schur!,
       phi1_t_e1_schur!,
       build_phi1_coeffs_schur,
       build_exp_coeffs_schur

using LinearAlgebra

# ============================================================
#  Stable 2x2 solver with SVD fallback
# ============================================================

@inline function solve2x2_stable(A11::Float64, A12::Float64,
                                A21::Float64, A22::Float64,
                                b1::Float64,  b2::Float64;
                                rcond::Float64 = 1e-14)

    detA = A11*A22 - A12*A21
    nA   = max(abs(A11)+abs(A12)+abs(A21)+abs(A22), 1.0)

    if abs(detA) > rcond * nA * nA
        invdet = 1.0 / detA
        x1 = ( A22*b1 - A12*b2) * invdet
        x2 = (-A21*b1 + A11*b2) * invdet
        return x1, x2
    else
        # Fallback: pseudoinverse via SVD
        A = [A11 A12; A21 A22]
        b = [b1; b2]

        F = svd(A)
        σmax = maximum(F.S)
        tol  = rcond * max(σmax, 1.0)

        y1 = (F.S[1] > tol) ? (dot(F.U[:,1], b) / F.S[1]) : 0.0
        y2 = (F.S[2] > tol) ? (dot(F.U[:,2], b) / F.S[2]) : 0.0

        x = F.Vt' * [y1; y2]
        return x[1], x[2]
    end
end


# ============================================================
#  Block back-substitution for real Schur matrices
# ============================================================

"""
    block_solve_quasitri!(z, T, b)

Solve T*z = b where T is real Schur (1x1 and 2x2 blocks).
Uses backward substitution with numerical safeguards.
"""
function block_solve_quasitri!(z::Vector{Float64},
                               T::Matrix{Float64},
                               b::Vector{Float64};
                               rcond::Float64 = 1e-14)

    k = size(T,1)
    fill!(z, 0.0)

    i = k
    while i >= 1
        if i > 1 && T[i, i-1] != 0.0
            # 2x2 block
            i1 = i - 1
            i2 = i

            rhs1 = b[i1]
            rhs2 = b[i2]

            if i2 < k
                @inbounds for j in (i2+1):k
                    rhs1 -= T[i1, j] * z[j]
                    rhs2 -= T[i2, j] * z[j]
                end
            end

            A11 = T[i1,i1]; A12 = T[i1,i2]
            A21 = T[i2,i1]; A22 = T[i2,i2]

            x1, x2 = solve2x2_stable(A11,A12,A21,A22, rhs1,rhs2; rcond=rcond)
            z[i1] = x1
            z[i2] = x2
            i -= 2
        else
            # 1x1 block
            rhs = b[i]
            if i < k
                @inbounds for j in (i+1):k
                    rhs -= T[i, j] * z[j]
                end
            end

            diag = T[i,i]
            τ = rcond * max(abs(diag), 1.0)
            denom = (abs(diag) > τ) ? diag : (diag == 0.0 ? τ : sign(diag)*τ)
            z[i] = rhs / denom
            i -= 1
        end
    end

    return z
end


# ============================================================
#  Schur cache
# ============================================================

struct SchurCache
    Q::Matrix{Float64}
    Tmat::Matrix{Float64}
    v::Vector{Float64}               # Q' * e1
    expT::Vector{Matrix{Float64}}    # optional cache of exp(t*T)
end


function build_schur_cache(Hs::Array{Float64,3};
                           ts::Union{Nothing,Vector{Float64}}=nothing)

    k, _, S = size(Hs)
    caches = Vector{SchurCache}(undef, S)

    for s in 1:S
        H = @view Hs[:,:,s]
        Sschur = schur(Matrix(H))
        Q = Matrix(Sschur.Z)
        T = Matrix(Sschur.T)
        v = copy(@view Q[:, 1])

        expT = Matrix{Float64}[]
        if ts !== nothing
            sizehint!(expT, length(ts))
            for t in ts
                push!(expT, exp(t .* T))
            end
        end

        caches[s] = SchurCache(Q, T, v, expT)
    end

    return caches
end


# ============================================================
#  exp(tH)e1 and t*phi1(tH)e1
# ============================================================

function exp_e1_schur!(out::Vector{Float64},
                       cache::SchurCache,
                       t::Float64;
                       expT_cached::Union{Nothing,Matrix{Float64}}=nothing)

    Q  = cache.Q
    T  = cache.Tmat
    v  = cache.v

    Et = (expT_cached === nothing) ? exp(t .* T) : expT_cached
    w  = Et * v
    mul!(out, Q, w)
    return out
end


function phi1_t_e1_schur!(out::Vector{Float64},
                          cache::SchurCache,
                          t::Float64;
                          expT_cached::Union{Nothing,Matrix{Float64}}=nothing,
                          rcond::Float64 = 1e-14)

    Q = cache.Q
    T = cache.Tmat
    v = cache.v

    if t == 0.0
        fill!(out, 0.0)
        return out
    end

    Et = (expT_cached === nothing) ? exp(t .* T) : expT_cached
    w  = Et * v
    b  = w .- v

    zT = similar(v)
    block_solve_quasitri!(zT, T, b; rcond=rcond)

    mul!(out, Q, zT)
    return out
end


# ============================================================
#  Coefficient builders
# ============================================================

function build_phi1_coeffs_schur(cache::Vector{SchurCache},
                                 times::Vector{Float64};
                                 rcond::Float64 = 1e-14,
                                 use_cache::Bool = true)

    S  = length(cache)
    k  = length(cache[1].v)
    Qn = length(times)

    coeffs = Array{Float64,3}(undef, k, Qn, S)
    tmp = zeros(Float64, k)

    for s in 1:S
        for (i, t) in enumerate(times)
            Et = (use_cache && length(cache[s].expT) == Qn) ? cache[s].expT[i] : nothing
            phi1_t_e1_schur!(tmp, cache[s], t; expT_cached=Et, rcond=rcond)
            @views coeffs[:, i, s] .= tmp
        end
    end

    return coeffs
end


function build_exp_coeffs_schur(cache::Vector{SchurCache},
                                deltas::Vector{Float64};
                                use_cache::Bool = true)

    S  = length(cache)
    k  = length(cache[1].v)
    Qn = length(deltas)

    coeffs = Array{Float64,3}(undef, k, Qn, S)
    tmp = zeros(Float64, k)

    for s in 1:S
        for (i, Δ) in enumerate(deltas)
            Et = (use_cache && length(cache[s].expT) == Qn) ? cache[s].expT[i] : nothing
            exp_e1_schur!(tmp, cache[s], Δ; expT_cached=Et)
            @views coeffs[:, i, s] .= tmp
        end
    end

    return coeffs
end

end # module
