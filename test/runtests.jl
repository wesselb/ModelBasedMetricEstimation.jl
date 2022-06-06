using FiniteDifferences
using Metrics
using Integrals
using Random
using Statistics
using ModelBasedMetricEstimation
using Test

Random.seed!(1)
@testset "ModelBasedMetricEstimation.jl" begin
    include("model.jl")
    include("estimate.jl")
end
