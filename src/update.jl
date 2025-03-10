#=******************************************************************************
Updating the model
******************************************************************************=#
export update_θ!





#-------------------------------------------------------------------------------
"""
    update_θ_vfi!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

Update the coefficients `mcl.θs` and `mcl.θEs` in the collocation model `mcl`
using fixed point iteration.

This opertion requires the user-supplied `mcl.Us` and `mcl.ΦXnext`.

This function should be called in each iteration of the solver.
"""
function update_θ_vfi!(mcl::MarkovCollocationUniform{D,N,K,J}) where {D,N,K,J}

    # intra-iteration stacking
    stackΦnext = blockdiag(mcl.ΦXnext...)

    # eval: RHS of the joint system
    rhs1 = vcat(
        vec(mcl.Us),
        spzeros(N*K)
    )
    rhs2 = blockdiag(
        mcl.β * stackΦnext,
        mcl.stackPz * mcl.stackΦXperm
    )
    coefOld = vcat(
        vec(mcl.θEs),
        vec(mcl.θs)
    )

    # eval: LHS of the joint system
    lhs = blockdiag(
        mcl.stackΦX,
        mcl.stackΦX
    )

    # solve: joint system
    coefNew = lhs \ Vector(rhs1 + rhs2 * coefOld)

    # update: θs and θEs
    mcl.θs = reshape(coefNew[N*K+1:end], N, K)
    mcl.θEs = reshape(coefNew[1:N*K], N, K)

    return nothing
end # update_θ_vfi!


#-------------------------------------------------------------------------------
"""
    update_θ_newton!(mcl::MarkovCollocationUniform{D,N,K,J})::SV64{K}

Update the coefficients `mcl.θs` and `mcl.θEs` in the collocation model `mcl`
using Newton's method.

This opertion requires the user-supplied `mcl.Us` and `mcl.ΦXnext`.
"""
function update_θ_newton!(
    mcl::MarkovCollocationUniform{D,N,K,J}
) where {D,N,K,J}

    # stack: old coefficient guess
    coefOld = vcat(
        vec(mcl.θs),
        vec(mcl.θEs)
    )

    # update: Jacobian's 2nd block
    mcl.Jacobian[1:N*K, N*K+1:end] = mcl.β * blockdiag(mcl.ΦXnext...)

    # eval: F(X)
    rhs1 = vcat(
        vec(mcl.Us),
        spzeros(N*K)
    )
    rhs2 = blockdiag(
        mcl.β * blockdiag(mcl.ΦXnext...),
        mcl.stackPz * mcl.stackΦXperm
    )
    rhs = rhs1 + rhs2 * vcat(
        vec(mcl.θEs),
        vec(mcl.θs)
    )
    lhs = blockdiag(
        mcl.stackΦX,
        mcl.stackΦX
    ) * coefOld

    # update: the stacked coefficientss
    coefNew = coefOld - mcl.Jacobian \ Vector(rhs - lhs)

    # update: θs and θEs
    mcl.θs = reshape(coefNew[N*K+1:end], N, K)
    mcl.θEs = reshape(coefNew[1:N*K], N, K)

    return nothing
end # update_θ_newton!





#-------------------------------------------------------------------------------
"""
    update_θ!(
        mcl::MarkovCollocationUniform{D,N,K,J} ;
        solver::Symbol = :fvi,
    ) where {D,N,K,J}

Using the user-supplied `mcl.Us` and `mcl.ΦXnext`, update the coefficients
`mcl.θs` and `mcl.θEs` in the collocation model `mcl`.

The solver can be either `:fvi` (fixed point iteration) or `:newton` (Newton's
method).

This function should be called in each iteration of the solver.
"""
function update_θ!(
    mcl::MarkovCollocationUniform{D,N,K,J} ;
    solver::Symbol = :fvi,
) where {D,N,K,J}

    if solver == :fvi
        update_θ_vfi!(mcl)
    elseif solver == :newton
        update_θ_newton!(mcl)
    else
        error("Unknown solver: $solver")
    end

    return nothing
end # update_θ!



