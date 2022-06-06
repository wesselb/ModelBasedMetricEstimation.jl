struct StudentsT <: Distribution
    StudentsT(;
        ν::Union{Nothing, Float64}=nothing,
        ε_ν::Union{Nothing, Float64}=isnothing(ν) ? nothing : ν / 2
    ) = isnothing(ν) ? new() : (
        new(),
        [_studentst_θ(ν)],
        [ε_ν / abs(_studentst_ν′(_studentst_θ(ν)))]
    )
end

_studentst_θ(ν) = log(99 / (ν - 1) - 1)
_studentst_ν(θ) = 1 + 99 / (1 + exp(θ[1]))
_studentst_ν′(θ) = -_studentst_ν(θ) * exp(θ[1]) / (1 + exp(θ[1]))

_integrand(z, ν) = (1 + z^2 / ν)^(-(ν + 1)/2)
_integrand′(z, ν) = Zygote.gradient(ν′ -> _integrand(z, ν′), ν)[1]

_tfun(x, ν) = _tfun(promote(x, ν)...)
_tfun(x::Float64, ν::Float64) =
    solve(
        IntegralProblem(_integrand, -Inf, x, ν),
        QuadGKJL(), reltol=1e-8
    ).u
function _tfun(x::Dual{T}, ν::Dual{T}) where T
    derivative_ν = solve(
        IntegralProblem(_integrand′, -Inf, value(x), value(ν)),
        QuadGKJL(), reltol=1e-8
    ).u
    derivative_x = _integrand(value(x), value(ν))
    return Dual{T}(
        _tfun(value(x), value(ν)), 
        derivative_x * partials(x) + derivative_ν * partials(ν)
    )
end

function logpdf(::StudentsT, x, θ)
    ν = _studentst_ν(θ)
    log_Z = log(ν * π) / 2 + loggamma(ν / 2) - loggamma((ν + 1) / 2)
    return -((ν + 1) / 2) .* log.(1 .+ x.^2 ./ ν) .- log_Z
end

function cdf(::StudentsT, x, θ)
    ν = _studentst_ν(θ)
    Z = sqrt(ν * π) * gamma(ν / 2) / gamma((ν + 1) / 2)
    return _tfun.(x, ν) / Z
end

rand(d::StudentsT, n, θ) = rand(Distributions.TDist(_studentst_ν(θ)), n)

function display(d::StudentsT, θ, ε, indent="")
    ν = _studentst_ν(θ)
    ε_ν = abs(_studentst_ν′(θ)) * ε[1]
    return indent * "StudentsT(ν=$(_round(ν)), ε_ν=$(_round(ε_ν)))\n"
end
