module NeoAlgebra
using SparseArrays
using LinearAlgebra
using Arpack
using Revise
using BenchmarkTools
using Printf
using Random
using KrylovKit
using IterativeSolvers
using NLsolve
include("Davidson.jl")
include("Benchmark.jl")
include("DIIS.jl")
export Davidson
export benchmark
export benchmark_diis
export run_chem_diis
export DIISManager
export diis_update!
export gdiis_update!
end
