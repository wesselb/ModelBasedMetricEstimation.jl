using TailPickles
using Distributions
using Metrics
using Statistics
using ProgressMeter

import TailPickles: Mixture, Gaussian, StudentsT, Location, Scale, Asymmetric

miso_ef = Mixture(
    Asymmetric(
        Gaussian() |> Scale(σ=8029.064, ε_σ=2709.973),
        Gaussian() |> Scale(σ=8039.926, ε_σ=2872.537)
    ) |> Location(μ=4091.623, ε_μ=2832.664),
    Asymmetric(
        StudentsT(ν=3.115, ε_ν=1.808) |> Scale(σ=17890.479, ε_σ=4774.823),
        StudentsT(ν=3.049, ε_ν=1.523) |> Scale(σ=17423.810, ε_σ=4285.292)
    ) |> Location(μ=4652.298, ε_μ=2377.236),
    π=[0.375, 0.625],
    ε_π=[0.161, 0.161]
)

pjm_ef = Mixture(
    Asymmetric(
        Gaussian() |> Scale(σ=4186.017, ε_σ=1691.401),
        Gaussian() |> Scale(σ=12045.631, ε_σ=2052.207)
    ) |> Location(μ=530.782, ε_μ=1820.250),
    Asymmetric(
        StudentsT(ν=2.549, ε_ν=1.091) |> Scale(σ=17265.067, ε_σ=3602.135),
        StudentsT(ν=2.075, ε_ν=0.996) |> Scale(σ=15251.450, ε_σ=3012.148)
    ) |> Location(μ=4604.778, ε_μ=1695.855),
    π=[0.421, 0.579],
    ε_π=[0.150, 0.150]
)

miso_gpf = Mixture(
    Asymmetric(
        Gaussian() |> Scale(σ=4531.472, ε_σ=4921.425),
        Gaussian() |> Scale(σ=6495.905, ε_σ=5957.561)
    ) |> Location(μ=-14116.721, ε_μ=5723.586),
    Asymmetric(
        Gaussian() |> Scale(σ=3504.921, ε_σ=2294.917),
        Gaussian() |> Scale(σ=8036.793, ε_σ=3542.341)
    ) |> Location(μ=-2462.058, ε_μ=2480.477),
    Asymmetric(
        StudentsT(ν=2.549, ε_ν=1.302) |> Scale(σ=17978.249, ε_σ=4217.318),
        StudentsT(ν=2.149, ε_ν=1.054) |> Scale(σ=20151.546, ε_σ=4987.616)
    ) |> Location(μ=-183.785, ε_μ=2738.765),
    π=[0.070, 0.192, 0.738],
    ε_π=[0.069, 0.148, 0.160]
)

pjm_gpf = Mixture(
    Asymmetric(
        Gaussian() |> Scale(σ=4273.118, ε_σ=2885.711),
        Gaussian() |> Scale(σ=5695.936, ε_σ=2167.965)
    ) |> Location(μ=908.154, ε_μ=1952.027),
    Asymmetric(
        StudentsT(ν=2.694, ε_ν=1.170) |> Scale(σ=17499.223, ε_σ=3153.008),
        StudentsT(ν=2.107, ε_ν=0.845) |> Scale(σ=18079.973, ε_σ=3115.628)
    ) |> Location(μ=1715.195, ε_μ=2086.636),
    π=[0.173, 0.827],
    ε_π=[0.154, 0.154]
)

ercot_ef = Mixture(
    Asymmetric(
        Gaussian() |> Scale(σ=76.885, ε_σ=43.830),
        Gaussian() |> Scale(σ=252.301, ε_σ=84.448)
    ) |> Location(μ=79.128, ε_μ=60.687),
    Asymmetric(
        StudentsT(ν=1.167, ε_ν=1.150) |> Scale(σ=425.808, ε_σ=81.788),
        StudentsT(ν=2.409, ε_ν=0.919) |> Scale(σ=541.552, ε_σ=85.153)
    ) |> Location(μ=343.604, ε_μ=58.362),
    π=[0.219, 0.781],
    ε_π=[0.180, 0.180]
)

bases = [
    ("MISO_EF", miso_ef),
    ("PJM_EF", pjm_ef),
    ("PJM_GPF", pjm_gpf),
    # ("ERCOT_EF", ercot_ef)
]

desired_mean_diffs = [
    0.00,
    250.0,
    250.0,
    500.0,
    1_000.0,
    2_000.0
]
desired_es_diffs = [
    10_000.0,
    10_000.0,
    0.00,
    20_000.0,
    30_000.0,
    50_000.0
]
base_es = 30_000.0
base_mean = 3_000.0

function tune_base_pairs(name, base1, base2, desired_mean_diff, desired_es_diff)
    d₁, θ₁, ε_θ₁ = base1
    d₂, θ₂, ε_θ₂ = base2
    
    # Tune first distribution.
    μ₁ = TailPickles.expectation(identity, d₁, θ₁)
    d₁, θ₁, ε_θ₁ = (d₁, θ₁, ε_θ₁) |> Location(μ=-μ₁)  # Set mean to zero.
    es₁ = TailPickles.es(d₁, 0.05, θ₁)
    d₁, θ₁, ε_θ₁ = (d₁, θ₁, ε_θ₁) |> Scale(σ=(base_es + base_mean) / es₁) |> Location(μ=base_mean) # Fix ES.
    
    # Tune second distribution.
    μ₂ = TailPickles.expectation(identity, d₂, θ₂)
    d₂, θ₂, ε_θ₂ = (d₂, θ₂, ε_θ₂) |> Location(μ=-μ₂)  # Set mean to zero.
    es₂ = TailPickles.es(d₂, 0.05, θ₂)
    d₂, θ₂, ε_θ₂ =
        (d₂, θ₂, ε_θ₂) |>
        Scale(σ=(base_es + desired_es_diff + base_mean + desired_mean_diff) / es₂) |>   # Set ES difference.
        Location(μ=base_mean + desired_mean_diff)  # Set mean difference.
    
    name = name * "/Δμ=$desired_mean_diff/ΔES=$desired_es_diff"
    return (name, (d₁, θ₁, ε_θ₁), (d₂, θ₂, ε_θ₂))
end

for (desired_mean_diff, desired_es_diff) in zip(desired_mean_diffs, desired_es_diffs)
    for (name1, base1) in bases, (name2, base2) in bases
        name, base1, base2 = tune_base_pairs("$name1/$name2", base1, base2, desired_mean_diff, desired_es_diff)
        d₁, θ₁, ε_θ₁ = base1
        d₂, θ₂, ε_θ₂ = base2
        μ₁ = TailPickles.expectation(identity, d₁, θ₁)
        μ₂ = TailPickles.expectation(identity, d₂, θ₂)
        es₁ = TailPickles.es(d₁, 0.05, θ₁)
        es₂ = TailPickles.es(d₂, 0.05, θ₂)
        println("Name: ", name)
        println("Δμ:   ", μ₂ - μ₁)
        println("Δes:  ", es₂ - es₁)
        @assert isapprox(μ₁, base_mean, rtol=1e-3, atol=1e-3)
        @assert μ₂ ≥ base_mean - 1e-3
        @assert isapprox(es₁, base_es, rtol=1e-3, atol=1e-3)
        @assert es₂ ≥ base_es - 1e-3
        @assert isapprox(μ₂ - μ₁, desired_mean_diff, rtol=1e-3, atol=1e-3)
        @assert isapprox(es₂ - es₁, desired_es_diff, rtol=1e-3, atol=1e-3)
    end
end