module TailPickles

using AdvancedHMC
using CSV
using DataFrames
using Dates
using Distributions
using FixedPointDecimals
using ForwardDiff
using Intervals
using JLSO
using LinearAlgebra
using Metrics
using ProgressMeter
using PyCall
using PyPlot
using Printf
using Quadrature
using SpecialFunctions
using Statistics
using StatsBase
using Zygote

import ForwardDiff: Dual, value, partials
import Base: rand

export estimate_metric

include("util.jl")
include("data.jl")
include("ci.jl")
include("model/model.jl")
include("model/students_t.jl")
include("model/laplace.jl")
include("model/gaussian.jl")
include("model/location.jl")
include("model/scale.jl")
include("model/linear.jl")
include("model/asymmetric.jl")
include("model/pareto.jl")
include("model/mixture.jl")
include("estimate.jl")

end