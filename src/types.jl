# export istransmat
export MarkovCollocationUniform



# ------------------------------------------------------------------------------
function speye(n::Int)::SpM64
    return SparseMatrixCSC{Float64,Int}(I, n, n)
end
# ------------------------------------------------------------------------------
"""
    istransmat(P::AbsM ; atol::F64 = 1E-6)::Bool

Check if a matrix `P` is a transition matrix at a given tolerance level `atol`.
"""
function istransmat(P::AbsM ; atol::F64 = 1E-6)::Bool

    if !(eltype(P) <: Real)
        return false
    end

    m,n = size(P)
    
    if m != n
        return false
    end
    if (m < 1) || isinf(m)
        return false
    end

    rowSum = sum(P, dims=2)
    if any(isinf.(rowSum)) || any(isnan.(rowSum))
        return false
    end
    if !all(isapprox.(rowSum, 1, atol = atol))
        return false
    end

    return true
end # istransmat





# ------------------------------------------------------------------------------
"""
    UniformMarkovCollocation{D,N,K,J}

A type for the collocation model on a Bellman equation with a `J` dimensional 
Markov uncertainty. Here the "uniform" means that the same basis matrix (or say
interpolation) of the endogenous states `x` applies to all `K` states of the 
Markov chain, but not essentially imply that the grid structure of `x` is the
Cartesian/Tensor-joined uniform grid.

v(x,z) = u(x,z) + β * E[v(x',z') | z]

E[v(x,z') | z] = ∑_{k=1}^K Pr{z'|z} * v(x,z')

where `x` is the `D`-dimensional endogenous state, `z` is `J`-dim exogenous 
state (shock) that follows a `K`-state Markov chain. The function `u` is the
instantaneous utility function, `β` is the discount factor, and `E` is the
expectation operator.

This structure is used for the collocation method to solve the Bellman equation,
but does NOT store the solution value function `v` itself. Please check another
package `FuncMarkov.jl` for the solution storage.

## Fields

- `ΦX::SpM64` : basis matrix at the grid nodes of the endogenous states `x`, the
size is `(N, N)` where `N` is the number of grid nodes
- `Pz::M64`   : transition matrix of the Markov chain for the exog states `z`,
the size is `(K, K)` where `K` is the number of states
- `dimnames::SizedVector{D,Sym}` : names of the states `x` dimension, the length
is `D`
- `β::F64` : discount factor

- `θs::Matrix{F64}` : coefficients accompanying the basis matrix `ΦX`, each col
is the coefficients for the value function at each state of the Markov chain. It
is initialized as zeros. The size is `(N, K)`.
- `θEs::Matrix{F64}` : coefficients accompanying the basis matrix `ΦX` for the
expected value function at each state of the Markov chain. The size is `(N, K)`.
It is initialized as zeros.
- `Jacobian::SpM64` : the Jacobian matrix for the collocation method. The size
is `(2*N*K, 2*N*K)`. It is initialized as zeros.

- `Us::Matrix{F64}` : the utility function `u(x,z)`, the size is `(N, K)`. It is
initialized as zeros.
- `ΦXnext::SizedVector{K,Matrix{F64}}` : the basis matrix `ΦX` at next period's
endo states for each state of the Markov chain. The size is `(N, N)` for each
elementary matrix. It is initialized as zeros.

- `stackΦX::SpM64` : the stacked basis matrix `ΦX` for the collocation method.
The size is `(N*K, N*K)`. It is initialized as zeros.
- `stackΦXperm::SpM64` : the stacked basis matrix `ΦX` permuted for the 
collocation method. Used in the expectation equation. The size is `(N*K, N*K)`.
It is initialized as zeros.
- `stackPz::M64` : the stacked transition matrix `Pz` for the collocation method
The size is `(N*K, N*K)`. It is initialized as zeros. Used in the expectation eq


## Constructor

    MarkovCollocationUniform(
        ΦX ::AbsM ,
        Pz ::AbsM ;
        D  ::I64 = 1,
        J  ::I64 = 1,
        dimnames::Union{Iterable,Nothing} = nothing,
        β       ::F64 = 0.95,
    )

## Example
```julia
col = include("src/Collocations.jl")
varinfo(col)

# ------------------------------------------------------------------------------
# initialize the interpolation basis matrix ΦX, transition matrix Pz
ΦX = col.sprand(50,50, 2/50); ΦX += col.I(50)
Pz = rand(2,2); Pz ./= sum(Pz, dims=2)

# ------------------------------------------------------------------------------
# define the collocation model
mcl = col.MarkovCollocationUniform(
    ΦX, Pz, 
    D = 5, J = 2,
    dimnames = ("a","b","c","d","e")
)

# ------------------------------------------------------------------------------
# load the initial guess of θs, θEs
mcl.θs  = rand(50,2)
mcl.θEs = rand(50,2)

# ------------------------------------------------------------------------------
# then, do the pre-conditioning procedures
col.precond!(mcl)

# validate the model after pre-conditioning
col.validate!(mcl)

# ------------------------------------------------------------------------------
# Usage: Fixed point iteration
mcl.ΦXnext = [ΦX for _ in 1:2]

col.update_θ!(mcl, solver = :fvi)
col.update_θ!(mcl, solver = :newton)



```
"""
mutable struct MarkovCollocationUniform{D,N,K,J}
    
    # fields: bricks & meta ----------------------------------------------------
    ΦX       ::SpM64 # size: (N,N)
    Pz       ::M64   # size: (K,K)
    dimnames ::SizedVector{D,Sym}
    β        ::F64

    # fields: coefficients (update by iteration) -------------------------------
    θs       ::Matrix{F64} # size: (N,K)
    θEs      ::Matrix{F64} # size: (N,K)
    Jacobian ::SpM64       # size: (2*N*K, 2*N*K)

    # fields: user-supplied iteration-by-iteration variables -------------------
    Us      ::Matrix{F64}          # size: (N,K)
    ΦXnext  ::SizedVector{K,SpM64} # element size: (N,N)

    # fields: pre-conditioner (no update by iteration) -------------------------
    stackΦX     ::SpM64 # size: (N*K, N*K)
    stackΦXperm ::SpM64 # size: (N*K, N*K)
    stackPz     ::SpM64 # size: (N*K, N*K)

    # constructor --------------------------------------------------------------
    function MarkovCollocationUniform(
        ΦX ::AbsM ,
        Pz ::AbsM ;
        D  ::I64 = 1,
        J  ::I64 = 1,
        dimnames::Union{Iterable,Nothing} = nothing,
        β       ::F64 = 0.95,
    )
        N = size(ΦX, 1)
        K = size(Pz, 1)
        @assert N == size(ΦX,2) "ΦX must be square for interpolation"
        @assert K == size(Pz,2) "Pz must be a squared transition matrix"
        @assert istransmat(Pz) "Pz must be a transition matrix"
        @assert length(dimnames) == D "dimnames must be of length $D"

        @assert (β > 0) & isfinite(β) "β must be positive and finite"


        _xnames = if isnothing(dimnames)
            [Symbol("x$i") for i in 1:D]
        else
            Symbol.(dimnames |> collect)
        end

        new{D,N,K,J}(
            SparseMatrixCSC{F64,I64}(ΦX), # ΦX
            Matrix{F64}(Pz),              # Pz
            SizedVector{D,Sym}(_xnames),  # dimnames
            β,                            # β

            zeros(F64,N,K), # θs
            zeros(F64,N,K), # θEs
            spzeros(2*N*K, 2*N*K), # Jacobian

            zeros(F64,N,K), # Us
            SizedVector{K,Matrix{F64}}([speye(N) for _ in 1:K]), # ΦXnext

            spzeros(N*K, N*K), # stackΦX
            spzeros(N*K, N*K), # stackΦXperm
            spzeros(N*K, N*K), # stackPz
        )
    end # constructor
end # UniformMarkovCollocation
# ------------------------------------------------------------------------------
function Base.show(io::IO, m::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    println(io, "UniformMarkovCollocation{D=$D,N=$N,K=$K,J=$J}")
    
    println(io, "- dimnames of x states        : $(m.dimnames)")
    println(io, "- dimension of endog states   : D = $D")
    println(io, "- dimension of exog shocks    : J = $J")
    println(io, "- discount factor             : β = $(m.β)")

    println(io, "- # of grid nodes             : N = $N")
    println(io, "- # of states in Markov chain : K = $K")
    
    println(io, "- # of v(x,z)       interp coefficients N*K = $(N*K)")
    println(io, "- # of E{v(x,z')|z} interp coefficients N*K = $(N*K)")
    
    println(io, "- size of Jacobian matrix : $(size(m.Jacobian))")
    return nothing
end








#=******************************************************************************
VALIDATION METHODS
******************************************************************************=#
"""
    validate!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Checks:

1. `ΦX` is invertible
2. `ΦXnext` are all invertible


Throw errors if any of the above conditions are not met. Returns nothing
"""
function validate!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    
    if rank(mcl.ΦX) != N
        error("ΦX is not invertible")
    end

    for k in 1:K
        if rank(mcl.ΦXnext[k]) != N
            error("ΦXnext[$k] is not invertible")
        end
    end


    return nothing
end # validate









