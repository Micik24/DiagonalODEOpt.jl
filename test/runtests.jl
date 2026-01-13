using Test
using DiagonalODEOpt

@testset "DiagonalODEOpt.jl" begin
    include("test_parametrization.jl")
    include("test_timegrid.jl")
    include("test_optimizer_adam.jl")
    include("test_adjoint_shapes.jl")
end
