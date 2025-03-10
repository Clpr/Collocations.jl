# export spblkdiag, spblkdiag_row, spblkdiag_col

export precond!



#=******************************************************************************
LINEAR ALGEBRA HELPERS
******************************************************************************=#
"""
    spblkdiag(X::AbsM{T}, outer::Int)::SpM{T}

Construct a block-diagonal matrix from a square matrix `X` by repeating the 
matrix `outer` times along the diagonal.

Returns a sparse matrix of size `(size(X, 1) * outer, size(X, 2) * outer)`.

## Example
```julia
using SparseArrays
col = include("src/Collocations.jl")

X = rand(3, 3)
spblkdiag(X, 2)
```
"""
function spblkdiag(X::AbsM{T}, outer::Int)::SpM{T} where T
    N, P = size(X)
    res = spzeros(N*outer, P*outer)
    for i in 1:outer
        res[(i-1)*N+1:i*N, (i-1)*P+1:i*P] = X
    end
    return res
end # blkdiag
# ------------------------------------------------------------------------------
"""
    spblkdiag_row(X::AbsM{T})::SpM{T} where T

Creates a sparse block diagonal by spliting the rows of `X` into blocks of size
`size(X, 1)` and stacking them along the diagonal. e.g.

## Example 
```
col = include("src/Collocations.jl")

X = [1 2; 3 4; 5 6]
res = col.spblkdiag_row(X)
```
"""
function spblkdiag_row(X::AbsM{T})::SpM{T} where T
    n,k = size(X)
    res = spzeros(T, n, k*n)
    for i in 1:n
        res[i, (i-1)*k+1:i*k] = X[i,:]
    end
    return res
end
# ------------------------------------------------------------------------------
"""
    spblkdiag_col(X::AbsM{T})::SpM{T} where T

Creates a sparse block diagonal by spliting the columns of `X` into blocks of
size `size(X, 2)` and stacking them along the diagonal. e.g.

## Example
```
col = include("src/Collocations.jl")

X = [1 2; 3 4; 5 6]
res = col.spblkdiag_col(X)
```
"""
function spblkdiag_col(X::AbsM{T})::SpM{T} where T
    n,k = size(X)
    res = spzeros(T, n*k, k)
    for i in 1:k
        res[(i-1)*n+1:i*n, i] = X[:,i]
    end
    return res
end












#=******************************************************************************
STACKING: COLLOCATION COMPONENTS

--------------------------------------------------------------------------------
Bellman equation:

stack(ΦX) * stack(θs) = stack(Us) + β * stack(ΦXnext) * stack(θEs)

where:

- Pre-conditionable:
    - stack(ΦX)
- Update by iteration (auto)
    - stack(θs)
    - stack(θEs)
- Update by iteration (user-suplied, from optimization in each iteration)
    - stack(Us)
    - stack(ΦXnext)

--------------------------------------------------------------------------------
Expectation equation:

stack(ΦX) * stack(θEs) = stack(Pz) * stack(ΦXperm) * stack(θs)

where:

- Pre-conditionable:
    - stack(ΦX)
    - stack(ΦXperm)
    - stack(Pz)
- Update by iteration (auto)
    - stack(θs)
    - stack(θEs)


******************************************************************************=#






#=******************************************************************************
STANDARD PRE-CONDITIONERS
******************************************************************************=#
"""
    precond_ΦX!(mcl::MarkovCollocationUniform{D,N,K,J})::SpM64 where {D,N,K,J}

Stack the `stack(ΦX)` coefficient matrix for the stacking Bellman equations. 
This is a standard pre-conditioning step for the collocation method.
"""
function precond_ΦX!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    for i in 1:K
        mcl.stackΦX[(i-1)*N+1:i*N, (i-1)*N+1:i*N] = mcl.ΦX
    end
    return nothing
end
# ------------------------------------------------------------------------------
"""
    precond_Pz!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Stack the `stack(Pz)` coefficient matrix for the stacking expectation equations.
This is a standard pre-conditioning step for the collocation method.
"""
function precond_Pz!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    ptr = 1
    for k in 1:K
        mcl.stackPz[ptr:ptr+N-1, :] = spblkdiag(mcl.Pz[k:k, :] |> sparse, N)
        ptr += N
    end
    return nothing
end
# ------------------------------------------------------------------------------
"""
    precond_ΦXperm!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Stack the `stack(ΦXperm)` coefficient matrix for the stacking expectation 
equations. This is a standard pre-conditioning step for the collocation method.

The permutation matrix is implicitly applied so no need to explicitly make it.
"""
function precond_ΦXperm!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    for i in 1:N
        # N vertical blocks; in each block, diagonal elements: {ϕ(x^i|z^k)}_k
        # under the uniform assumption, ϕ(x^i) for K times in the diagonal.
        rowHead = (i-1)*K+1
        rowTail = i*K

        mcl.stackΦXperm[rowHead:rowTail, :] = spblkdiag(
            mcl.ΦX[i:i, :] |> sparse, 
            K
        )
    end # i
    return nothing
end


# ------------------------------------------------------------------------------
"""
    precond_jacobian!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Construct the Jacobian matrix for the collocation method. This is a standard
pre-conditioning step for the collocation method.

IMPORTANT: must pre-condition `stack_ΦX`, `stack_Pz`, `stack_ΦXperm` before
calling this function. This function uses the pre-conditioned matrices to
avoid duplicate computation.

NOTE: not like `stackΦX`, `stackPz`, `stackΦXperm`, the `Jacobian` matrix will
be updated by iteration **partially** (only a block will be updated).

NOTE: the pseudo linear system is defined as RHS - LHS of the Bellman equation
and the expectation equation.

Jacobian = [
    1st block, 2nd block;
    3rd block, 4th block
]
where the 2nd block depends on stackΦXnext which is updated by iteration.
"""
function precond_jacobian!(
    mcl::MarkovCollocationUniform{D,N,K,J}
) where {D,N,K,J}

    #=
    Jacobian = [
        1st block, 2nd block;
        3rd block, 4th block
    ] = [
        -stackΦX            ,  β*stackΦXnext;
        stackPz*stackΦXperm ,  -stackΦX
    ]

    Each block is (NK,NK) size
    =#


    # 1st block: - stack(ΦX)
    mcl.Jacobian[1:N*K, 1:N*K] = -mcl.stackΦX

    # 4th block: - stack(ΦX)
    mcl.Jacobian[N*K+1:end, N*K+1:end] = -mcl.stackΦX

    # 2nd block: 
    mcl.Jacobian[1:N*K, N*K+1:end] = mcl.β * blockdiag(mcl.ΦXnext...)

    # 3rd block:
    mcl.Jacobian[N*K+1:end, 1:N*K] = mcl.stackPz * mcl.stackΦXperm

    return nothing
end


# ------------------------------------------------------------------------------
"""
    precond!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Run all the standard pre-conditioning procedures collectively in order:

1. `precond_ΦX!`
2. `precond_Pz!`
3. `precond_ΦXperm!`
4. `precond_jacobian!`

use this function primarily after constructing a `MarkovCollocationUniform`
object to pre-condition the collocation components.
"""
function precond!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}
    precond_ΦX!(mcl)
    precond_Pz!(mcl)
    precond_ΦXperm!(mcl)
    precond_jacobian!(mcl)
    return nothing
end
















