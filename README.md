# DiagonalODEOpt.jl

**GPU-accelerated AI-driven optimization of diagonal
parameters in linear ODE systems using adjoint Krylov methods.**

This package provides an efficient framework for optimizing diagonal parameters
of large-scale linear ordinary differential equations of the form:

$$
x'(t) = (A₀ + \mathrm{diag}(d(θ)))\,x(t) + f
$$

where:

- **$A₀ \in \mathbb{R}^{n \times n}$** is a fixed base operator,
- **$θ \in \mathbb{R}^n$** are trainable diagonal parameters,
- **$d(θ)$** is a monotone parametrization enforcing stability,
- **$f \in \mathbb{R}^{n \times s}$** are constant forcing vectors (many right-hand sides).

The implementation combines:

- GPU-resident batched Krylov / Arnoldi projections,
- CPU small-matrix Schur decompositions for stable matrix functions,
- Adjoint sensitivity analysis for gradient computation,
- First-order optimizers (Adam) for training diagonal parameters.

The code is designed for research, numerical experimentation, and HPC-oriented workflows.

---

## Key Features

- Fully GPU-based batched Arnoldi / Krylov projections.
- Stable evaluation of `exp(tH)` and `t·φ₁(tH)` via real Schur decomposition.
- Adjoint-based gradients with respect to diagonal parameters.
- Modular diagonal parametrizations (exp, softplus, or custom).
- Support for many trajectories / right-hand sides in parallel.
- Pluggable optimization layer (Adam provided, others easy to add).
- Automatic masking of numerically inactive trajectories.

---

## Dependency

This package depends on:

**Batched_Krylov_GPU**  
https://github.com/Micik24/Batched_Krylov_GPU

Batched_Krylov_GPU provides the GPU implementation of the batched Arnoldi method
used to construct Krylov bases and Hessenberg matrices.

This package does not duplicate that functionality and requires it as a dependency.

---

## Installation

### Requirements

- **Julia ≥ 1.10**
- **CUDA-capable GPU** with CUDA.jl properly configured
- **Git**

### Clone the repository

```bash
git clone https://github.com/<your-username>/DiagonalODEOpt.jl.git
cd DiagonalODEOpt.jl
```

### Start Julia and activate the local environment

```text
julia
]
activate .
```

### Install the Krylov dependency

```text
add https://github.com/Micik24/Batched_Krylov_GPU
```

### Instantiate and precompile

```text
instantiate
precompile
```

### Verify installation

```julia
using DiagonalODEOpt
using CUDA
```

---

## Mathematical model

We consider a linear ODE system:

$$
x'(t) = A x(t) + f, \qquad
A = A₀ + \mathrm{diag}(d(θ)),
$$

with constant forcing $f$.

The exact solution at time $T$ is:

$$
x(T) = \exp(TA)x(0) + T \cdot φ₁(TA) f
$$

where:

$$
φ₁(A) = A^{-1}(\exp(A) - I).
$$

The action of these operators is approximated using Krylov subspaces
constructed via batched Arnoldi iterations on GPU.

---

## Objective and gradient

A generic loss $L(x(T))$ is evaluated at final time $T$
(e.g. cross-entropy, regression loss, or custom objective).

The gradient with respect to diagonal parameters is computed via an adjoint method:

$$
\frac{\partial L}{\partial θ}
= \int Λ(t) \odot X(t)\,dt \cdot \left(\frac{\partial d}{\partial θ}\right),
$$

where:

- **$Λ(t)$** is the adjoint solution,
- **$X(t)$** is the forward forcing contribution,
- **$\odot$** denotes elementwise multiplication.

Time integration is performed using user-provided quadrature weights.

---

## Diagonal parametrization

The mapping from parameters $θ$ to diagonal entries $d(θ)$ is modular.

Two parametrizations are provided:

### ExpParam()

$$
d(θ) = -\exp(θ)
$$

### SoftplusParam()

$$
d(θ) = -\mathrm{softplus}(θ)
$$

Custom parametrizations can be added by implementing:

```julia
diag_from_theta(param, θ)
dtheta_from_ddiag(param, θ)
```

---

## Optimization

The package includes a GPU-safe implementation of Adam, used to optimize $θ$
based on adjoint gradients.

The optimization loop is intentionally decoupled from gradient computation:

- `step_grad_theta!` computes loss and gradient,
- optimizers update parameters externally.

This design allows easy experimentation with alternative optimizers.

---

## Usage example

```julia
using DiagonalODEOpt
using CUDA

CUDA.allowscalar(false)

n = 256      # state dimension
S = 64       # number of forcing vectors

A0 = CUDA.randn(Float64, n, n) .* 0.01
@views A0[diagind(A0)] .= 0.0

F = CUDA.randn(Float64, n, S)
labels = rand(1:n, S)

θ = CUDA.zeros(Float64, n)
param = SoftplusParam()

θ, history = optimize_diagonal_adam!(
    A0, θ, F, labels, param;
    T = 2.0,
    Q = 32,
    epochs = 20,
    lr = 1e-3,
    k = 30,
    β_soft = 10.0,
    batchsize = 32,
    clamp_theta = (-50.0, 50.0),
    callback = (ep, loss, θ) ->
        println("epoch $ep | loss = $(round(loss, sigdigits=6))")
)
```

---

## Numerical notes

- Krylov projections are performed entirely on GPU.
- Small Hessenberg matrices are processed on CPU using real Schur decomposition.
- $φ₁$ is evaluated via stable block back-substitution, not explicit inversion.
- Numerically inactive trajectories are automatically masked.
- All computations are performed in Float64 for stability.

---

## Repository structure

```text
src/
 ├── DiagonalODEOpt.jl      # package entry point
 ├── Parametrization.jl    # θ → diagonal mapping
 ├── AdjointStep.jl        # adjoint Krylov gradient
 ├── SchurCPU.jl           # small-matrix CPU numerics
 ├── ReconstructionGPU.jl # batched GPU reconstruction
 ├── Training.jl           # optimization driver
 └── Optimizers/
     └── Adam.jl           # Adam optimizer
examples/
 └── synthetic_demo.jl
```

---

## Limitations

- Only diagonal parametrizations are supported.
- Only Float64 precision is currently supported.
- The package is not registered in the Julia General registry.
- Intended primarily for research and advanced numerical experimentation.

---

## License

MIT

---

## Author

**Maciej Wajda**  
maciejwajda01@gmail.com
