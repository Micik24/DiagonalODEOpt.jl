using Test
using DiagonalODEOpt
using CUDA

CUDA.functional() || @info "CUDA not available, skipping GPU test"

@testset "Adjoint gradient shapes" begin
    if !CUDA.functional()
        @test true
        return
    end

    n = 16
    S = 4

    A0 = CUDA.randn(Float64, n, n) .* 0.01
    @views A0[diagind(A0)] .= 0.0

    F = CUDA.randn(Float64, n, S)
    θ = CUDA.zeros(Float64, n)
    labels = rand(1:n, S)

    param = SoftplusParam()

    times, weights = trapezoid_times_weights(1.0, 8)
    weights_gpu = CuArray(weights)

    loss, grad = step_grad_theta!(
        A0, θ, F, labels, param;
        T = 1.0,
        times = times,
        weights_gpu = weights_gpu,
        k = 10,
        β_soft = 5.0
    )

    @test isfinite(loss)
    @test size(grad) == size(θ)
    @test all(isfinite, Array(grad))
end
