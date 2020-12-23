using TailPickles
using Distributions
using Metrics
using Statistics
using ProgressMeter

import TailPickles: Gaussian, StudentsT, Location, Scale, Asymmetric

n = 500
reps = 1000
method = :laplace
laplace_sample = true
components = 1

for (d, θ, _) in [
    StudentsT(ν=2.0) |> Location(μ=5.0) |> Scale(σ=1.0),
    StudentsT(ν=5.0) |> Location(μ=5.0) |> Scale(σ=1.0),
    StudentsT(ν=10.0) |> Location(μ=5.0) |> Scale(σ=1.0),
    StudentsT(ν=20.0) |> Location(μ=5.0) |> Scale(σ=1.0),
    Gaussian() |> Location(μ=5.0) |> Scale(σ=1.0),
    TailPickles.Laplace() |> Location(μ=5.0) |> Scale(σ=1.0),
    Asymmetric(
        TailPickles.Laplace() |> Scale(σ=1.0),
        TailPickles.Laplace() |> Scale(σ=2.0),
    ) |> Location(μ=5.0),
    Asymmetric(
        TailPickles.Laplace() |> Scale(σ=4.0),
        TailPickles.Laplace() |> Scale(σ=1.0),
    ) |> Location(μ=5.0)
]
    print(TailPickles.display(d, θ, zeros(length(θ))))

    true_mean = TailPickles.expectation(identity, d, θ)
    true_median = TailPickles.icdf(d, 0.5, θ, x₀=true_mean)
    true_es = TailPickles.es(d, 0.05, θ, x₀=true_median)
    true_moe = true_mean / true_es
    true_medoe = true_median / true_es
    
    coverage_mean, coverage_mean_ci = TailPickles.ci(mean,
        @showprogress "Estimating coverage for mean: " [
            true_mean in estimate_metric(
                :mean,
                TailPickles.rand(d, n, θ),
                method=method,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
    )
    println("Coverage of mean: $(coverage_mean) $(coverage_mean_ci)")

    coverage_median, coverage_median_ci = TailPickles.ci(mean,
        @showprogress "Estimating coverage for median: " [
            true_median in estimate_metric(
                :median,
                TailPickles.rand(d, n, θ),
                method=method,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
    )
    println("Coverage of median: $(coverage_median) $(coverage_median_ci)")

    coverage_es, coverage_es_ci = TailPickles.ci(mean,
        @showprogress "Estimating coverage for ES: " [
            true_es in estimate_metric(
                :es,
                TailPickles.rand(d, n, θ),
                method=method,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
    )
    println("Coverage of ES: $(coverage_es) $(coverage_es_ci)")

    coverage_moe, coverage_moe_ci = TailPickles.ci(mean,
        @showprogress "Estimating coverage for mean/ES: " [
            true_moe in estimate_metric(
                :mean_over_es,
                TailPickles.rand(d, n, θ),
                method=method,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
    )
    println("Coverage of mean/ES: $(coverage_moe) $(coverage_moe_ci)")

    coverage_medoe, coverage_medoe_ci = TailPickles.ci(mean,
        @showprogress "Estimating coverage for median/ES: " [
            true_medoe in estimate_metric(
                :median_over_es,
                TailPickles.rand(d, n, θ),
                method=method,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
    )
    println("Coverage of median/ES: $(coverage_medoe) $(coverage_medoe_ci)")
end
