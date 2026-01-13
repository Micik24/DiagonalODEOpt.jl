module ReconstructionGPU

export reconstruct_times!

using CUDA
using Batched_Krylov_GPU: gemm_strided_batched!

CUDA.allowscalar(false)

"""
    reconstruct_times!(Y, Vall, coeffs, beta, active; k, kmax)

Compute:

    Y[:,i,s] = beta[s] * V_s * coeffs[:,i,s]

for all s and i using a single batched GEMM call.
Inactive samples are zeroed.
"""
function reconstruct_times!(
    Y::CuArray{Float64,3},
    Vall::CuArray{Float64,3},
    coeffs::CuArray{Float64,3},
    beta::CuArray{Float64,1},
    active::CuArray{Bool,1};
    k::Int,
    kmax::Int
)
    n, Q, S = size(Y)

    V = @view Vall[:, 1:k, :]       # n × k × S

    B = similar(coeffs)            # k × Q × S
    beta3  = reshape(beta, 1, 1, S)
    mask3  = reshape(active, 1, 1, S)

    B .= coeffs .* beta3
    B .= ifelse.(mask3, B, 0.0)

    strideV = n * (kmax + 1)
    strideB = k * Q
    strideC = n * Q

    gemm_strided_batched!(
        'N', 'N',
        1.0,
        V, n, strideV,
        B, k, strideB,
        0.0,
        Y, n, strideC,
        S
    )

    return Y
end

end # module
