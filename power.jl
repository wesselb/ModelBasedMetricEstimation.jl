using TailPickles
using Distributions
using Metrics
using Statistics
using ProgressMeter

import TailPickles: Gaussian, StudentsT, Location, Scale, Asymmetric

desired_mean_diff = 200_000.0 / 365
desired_es = 30_000.0
desired_es_diff = 10_000.0
reps = 1000
laplace_sample = false
components = 1

test_mean = true
test_es = true
test_moe = true

# Both heavily tailed to the left.
d₁, θ₁, ε_θ₁ = Asymmetric(StudentsT(ν=1.5), TailPickles.Gaussian() |> Scale(σ=1.0))
d₂, θ₂, ε_θ₂ = Asymmetric(
    TailPickles.Laplace() |> Scale(σ=1.0),
    TailPickles.Gaussian() |> Scale(σ=2.0)
)

# Tune first distribution.
μ₁ = TailPickles.expectation(identity, d₁, θ₁)
d₁, θ₁, ε_θ₁ = (d₁, θ₁, ε_θ₁) |> Location(μ=-μ₁)  # Set zero mean.
es₁ = TailPickles.es(d₁, 0.05, θ₁)
d₁, θ₁, ε_θ₁ = (d₁, θ₁, ε_θ₁) |> Scale(σ=desired_es / es₁)  # Fix ES.

# Tune second distribution.
μ₂ = TailPickles.expectation(identity, d₂, θ₂)
d₂, θ₂, ε_θ₂ = (d₂, θ₂, ε_θ₂) |> Location(μ=-μ₂)  # Set zero mean.
es₂ = TailPickles.es(d₂, 0.05, θ₂)
d₂, θ₂, ε_θ₂ =
    (d₂, θ₂, ε_θ₂) |>
    Scale(σ=(desired_es + desired_es_diff - desired_mean_diff) / es₂) |>   # Fix ES.
    Location(μ=-desired_mean_diff)  # Set mean difference.
    
# Compute exact differences.
mean_diff = (
    TailPickles.expectation(identity, d₁, θ₁) -
    TailPickles.expectation(identity, d₂, θ₂)
)
es_diff = (
    TailPickles.es(d₁, 0.05, θ₁) -
    TailPickles.es(d₂, 0.05, θ₂)
)
moe_diff = (
    TailPickles.expectation(identity, d₁, θ₁) / TailPickles.es(d₁, 0.05, θ₁) -
    TailPickles.expectation(identity, d₂, θ₂) / TailPickles.es(d₂, 0.05, θ₂)
)

println("Dist. 1:")
print(TailPickles.display(d₁, θ₁, zeros(length(θ₁)), "  "))
println("Dist. 2:")
print(TailPickles.display(d₂, θ₂, zeros(length(θ₂)), "  "))

println("Desired difference in mean:               $(desired_mean_diff)")
println("Desired difference in ES                  $(desired_es_diff)")
println("True difference in mean:                  $(mean_diff)")
println("True difference in ES:                    $(es_diff)")
println("True difference in mean/ES:               $(moe_diff)")

for n in 182 .* [2, 3, 4, 5, 6]
    println("n: ", n)
    samples_x = [TailPickles.rand(d₁, n, θ₁) for _ = 1:reps]
    samples_y = [TailPickles.rand(d₂, n, θ₂) for _ = 1:reps]

    ## Mean:

    if test_mean
        cis = @showprogress "Estimating power and coverage for mean: " [
            TailPickles.ci(
                mean,
                samples_x[i] .- samples_y[i]
            )[2]
            for i = 1:reps
        ]
        pow_mean, pow_mean_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_mean, cov_mean_ci = TailPickles.ci(mean, [mean_diff in ci for ci in cis])
        println("Naive:")
        println("  Coverage for difference in mean:          $(cov_mean) $(cov_mean_ci)")
        println("  Power for detecting a difference in mean: $(pow_mean) $(pow_mean_ci)")
            
        cis = @showprogress "Estimating power and coverage for mean: " [
            Metrics.subsample_difference_ci(
                Metrics.mean,
                samples_x[i],
                samples_y[i],
                β=0.5, sizemin=4, sizemax=180
            )
            for i = 1:reps
        ]
        pow_mean, pow_mean_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_mean, cov_mean_ci = TailPickles.ci(mean, [mean_diff in ci for ci in cis])
        println("Subsampling:")
        println("  Coverage for difference in mean:          $(cov_mean) $(cov_mean_ci)")
        println("  Power for detecting a difference in mean: $(pow_mean) $(pow_mean_ci)")
    
        cis = @showprogress "Estimating power and coverage for mean: " [
            estimate_metric(
                :mean,
                samples_x[i],
                samples_y[i],
                method=:laplace,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
        pow_mean, pow_mean_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_mean, cov_mean_ci = TailPickles.ci(mean, [mean_diff in ci for ci in cis])
        println("Parametric:")
        println("  Coverage for difference in mean:          $(cov_mean) $(cov_mean_ci)")
        println("  Power for detecting a difference in mean: $(pow_mean) $(pow_mean_ci)")
    end
    
    ## ES:

    if test_es
        cis = @showprogress "Estimating power and coverage for ES: " [
            Metrics.subsample_difference_ci(
                Metrics.es,
                samples_x[i],
                samples_y[i],
                β=0.5, sizemin=40, sizemax=180
            )
            for i = 1:reps
        ]
        pow_es, pow_es_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_es, cov_es_ci = TailPickles.ci(mean, [es_diff in ci for ci in cis])
        println("Subsampling:")
        println("  Coverage for difference in ES:          $(cov_es) $(cov_es_ci)")
        println("  Power for detecting a difference in ES: $(pow_es) $(pow_es_ci)")
    
        cis = @showprogress "Estimating power and coverage for ES: " [
            estimate_metric(
                :es,
                samples_x[i],
                samples_y[i],
                method=:laplace,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
        pow_es, pow_es_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_es, cov_es_ci = TailPickles.ci(mean, [es_diff in ci for ci in cis])
        println("Parametric:")
        println("  Coverage for difference in ES:          $(cov_es) $(cov_es_ci)")
        println("  Power for detecting a difference in ES: $(pow_es) $(pow_es_ci)")
    end

    ## MOE:

    if test_moe
        cis = @showprogress "Estimating power and coverage for mean/ES: " [
            Metrics.subsample_difference_ci(
                Metrics.mean_over_es,
                samples_x[i],
                samples_y[i],
                β=0.5, sizemin=40, sizemax=180
            )
            for i = 1:reps
        ]
        pow_moe, pow_moe_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_moe, cov_moe_ci = TailPickles.ci(mean, [moe_diff in ci for ci in cis])
        println("Subsampling:")
        println("  Coverage for difference in mean/ES:          $(cov_moe) $(cov_moe_ci)")
        println("  Power for detecting a difference in mean/ES: $(pow_moe) $(pow_moe_ci)")
    
        cis = @showprogress "Estimating power and coverage for mean/ES: " [
            estimate_metric(
                :mean_over_es,
                samples_x[i],
                samples_y[i],
                method=:laplace,
                laplace_sample=laplace_sample,
                components=components
            )[2]
            for i = 1:reps
        ]
        pow_moe, pow_moe_ci = TailPickles.ci(mean, [!(0 in ci) for ci in cis])
        cov_moe, cov_moe_ci = TailPickles.ci(mean, [moe_diff in ci for ci in cis])
        println("Parametric:")
        println("  Coverage for difference in mean/ES:          $(cov_moe) $(cov_moe_ci)")
        println("  Power for detecting a difference in mean/ES: $(pow_moe) $(pow_moe_ci)")
    end
end
