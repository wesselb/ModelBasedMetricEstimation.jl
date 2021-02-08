using FiniteDifferences
using Metrics
using Quadrature
using Statistics
using ModelBasedMetricEstimation
using Test

@testset "ModelBasedMetricEstimation.jl" begin
    include("model.jl")
    include("estimate.jl")
end
