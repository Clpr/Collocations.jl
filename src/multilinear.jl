#=******************************************************************************
A bare-bone implementation of multi-linear interpolation

# Assumptions
1. Evenly spaced grid points
2. Tensor product grid
3. No boundary treatment
4. Linear extrapolation
******************************************************************************=#





#===============================================================================
Math helpers
===============================================================================#
"""
    row_khatri_rao(Xs::AbsV{SpM64})::SpM64

Compute the row-wise Khatri–Rao product of a list of matrices `Xs`. The row-wise
Khatri–Rao product is defined as the row-wise Kronecker product of the matrices
in `Xs`.

Returns a sparse matrix of size `(n, m1*m2*...*mk)` where `n` is the number of
rows in each matrix in `Xs`, and `m1, m2, ..., mk` are the number of columns in
each matrix in `Xs`.

## Example
```julia
col = include("src/Collocations.jl")

# mimic: 1000-grid points total, 3 dim, 10 grid nodes per dim 
#        multi-linear interpolation
Xs = [sprand(1000,10, 2/100) for _ in 1:3]

# row-wise Khatri–Rao product (to build tensor product grid)
col.row_khatri_rao(Xs)
```
"""
function row_khatri_rao(Xs::AbsV{SpM64})::SpM64
    N = size(Xs[1], 1)
    @assert length(Xs) > 1 "At least two matrices are required"
    @assert all(size.(Xs, 1) .== N) "Matrices must have the same number of rows"

    Ms = size.(Xs, 2)

    Y = spzeros(N, prod(Ms))

    for i in 1:N
        Y[i, :] = kron([X[i, :] for X in Xs]...)
    end

    return Y
end # row_khatri_rao








#===============================================================================
Hat/Basis function
===============================================================================#
"""



"""




































