module DiagonalODEOpt

include("Parametrization.jl")
include("SchurCPU.jl")
include("ReconstructionGPU.jl")
include("AdjointStep.jl")
include("TimeGrid.jl")
include("Training.jl")
include("Optimizers/Adam.jl")

using .Parametrization
using .AdjointStep
using .TimeGrid
using .Training
using .AdamOptimizer

# ---- Re-exports: API użytkownika ----

export AbstractDiagParametrization, ExpParam, SoftplusParam
export diag_from_theta, dtheta_from_ddiag, softplus

export step_grad_theta!                  # gradient (adjoint Krylov)
export trapezoid_times_weights           # time grid helper

export AdamState, adam_init, adam_step!  # optimizer building blocks
export optimize_diagonal_adam!           # high-level training driver

end # module
