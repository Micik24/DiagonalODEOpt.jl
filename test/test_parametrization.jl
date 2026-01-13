using Test
using DiagonalODEOpt

@testset "Parametrization" begin
    θ = [-2.0, 0.0, 1.0]

    p1 = ExpParam()
    d1 = diag_from_theta(p1, θ)

    @test length(d1) == length(θ)
    @test all(d1 .< 0)

    p2 = SoftplusParam()
    d2 = diag_from_theta(p2, θ)

    @test length(d2) == length(θ)
    @test all(d2 .< 0)

    # chain rule sanity
    g = dtheta_from_ddiag(p2, θ)
    @test all(isfinite, g)
end
