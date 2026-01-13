using Test
using DiagonalODEOpt

@testset "Adam optimizer (CPU sanity)" begin
    θ = randn(5)
    g = ones(5)

    # fake AdamState on CPU (for logic test only)
    st = DiagonalODEOpt.AdamOptimizer.AdamState(
        zeros(5), zeros(5), 0, 1.0, 1.0
    )

    θ_old = copy(θ)

    DiagonalODEOpt.AdamOptimizer.adam_step!(
        θ, g, st;
        lr = 0.1,
        weight_decay = 0.0
    )

    @test θ != θ_old
    @test all(isfinite, θ)
end
