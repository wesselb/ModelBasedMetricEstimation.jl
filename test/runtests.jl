using FiniteDifferences
using Metrics
using Quadrature
using Statistics
using MBME
using Test

@testset "MBME.jl" begin
    include("model.jl")
    include("estimate.jl")
end
