module Collocations
# ==============================================================================
using LinearAlgebra, SparseArrays

using StaticArrays, NamedArrays



# ------------------------------------------------------------------------------
# alias: std elementary type
const F64 = Float64
const F32 = Float32
const I64 = Int64
const Str = String
const Sym = Symbol

# alias: std collections
const V64 = Vector{Float64}
const M64 = Matrix{Float64}
const V32 = Vector{Float32}
const M32 = Matrix{Float32}
const Dict64 = Dict{Symbol,Float64}
const NTup64{D} = NTuple{D,Float64}

const SpV64 = SparseVector{Float64}
const SpM64 = SparseMatrixCSC{Float64,Int64}
const SpM{T} = SparseMatrixCSC{T}

# alias: std abstract types
const AbsV = AbstractVector
const AbsM = AbstractMatrix
const AbsVM = AbstractVecOrMat

const Iterable = Union{AbsV, Tuple}

# alias: StaticArrays.jl
const SV64{D}   = SVector{D,Float64}
const SM64{D,K} = SMatrix{D,K,Float64}

# alias: NamedArrays.jl
const NmV64 = NamedVector{Float64}
const NmM64 = NamedMatrix{Float64}





# ------------------------------------------------------------------------------

# data type
include("types.jl")

# pre-conditioner
include("preconditioner.jl")

# update API
include("update.jl")




# ==============================================================================
end # module Collocations
