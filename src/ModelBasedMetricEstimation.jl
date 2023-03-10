module ModelBasedMetricEstimation

using AdvancedHMC
using ChainRulesCore: ignore_derivatives
using Distributions
using ForwardDiff
using Intervals
using LinearAlgebra
using Metrics
using NLopt
using Plots
using Printf
using ProgressMeter
using Integrals
using SpecialFunctions
using Statistics
using StatsBase
using Zygote

import Base: rand
import ForwardDiff: Dual, value, partials

export estimate_metric

include("util.jl")
include("model/model.jl")
include("model/students_t.jl")
include("model/laplace.jl")
include("model/gaussian.jl")
include("model/location.jl")
include("model/scale.jl")
include("model/linear.jl")
include("model/asymmetric.jl")
include("model/mixture.jl")
include("estimate.jl")

end
