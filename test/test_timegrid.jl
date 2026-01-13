using Test
using DiagonalODEOpt

@testset "Time grid" begin
    T = 2.0
    Q = 10

    times, weights = trapezoid_times_weights(T, Q)

    @test length(times) == Q + 1
    @test length(weights) == Q + 1

    @test times[1] ≈ 0.0
    @test times[end] ≈ T

    # trapezoidal rule integrates constant exactly
    integral = sum(weights)
    @test integral ≈ T
end
