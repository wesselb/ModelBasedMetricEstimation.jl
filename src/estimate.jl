"""
    lbfgs(target, θ₀; verbose=true)

Run L-BFGS to maximise a target function.
"""
function lbfgs(target, θ₀; verbose=true)
    target_min(θ) = -target(θ)  # Turn maximisation into a minimisation problem.
    objective(θ) = target_min(θ), ForwardDiff.gradient(target_min, θ)
    so = pyimport("scipy.optimize")
    return so.fmin_l_bfgs_b(
        func=objective,
        x0=θ₀,
        maxiter=20_000,
        maxfun=100_000,
        disp=Int(verbose),
        maxls=100,
        factr=10,
        pgtol=1e-8
    )[1]
end

"""
    hmc(target, θ₀, n; verbose=true)

Run HMC to sample from a target density.
"""
function hmc(target, θ₀, ε_θ₀, n; verbose=true)
    # Define Hamiltonian system.
    metric = DiagEuclideanMetric((ε_θ₀ ./ 1.96).^2)  # Error ≈ 2σ.
    hamiltonian = Hamiltonian(metric, target, ForwardDiff)

    # Construct integrator.
    ε₀ = find_good_stepsize(hamiltonian, θ₀)
    integrator = Leapfrog(ε₀)

    # Construct sampler.
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run sampler.
    samples, _ = sample(hamiltonian, proposal, θ₀, 2n, adaptor, n; progress=verbose)
    return samples[n + 1:end]  # Remove adaptation samples.
end

_make_target(x, prior, α, likelihood; features=nothing) =
    θ -> sum(logpdf(likelihood, x, θ, features=features)) + sum(logpdf(prior, θ, α))

function estimate_hmc(
    x,
    prior, α,
    likelihood, θ₀, ε_θ₀,
    n, m,
    functional;
    verbose=false,
    features=nothing,
    _return_samples=false
)
    target = _make_target(x, prior, α, likelihood, features=features)
    if verbose
        println("θ₀:")
        print(display(likelihood, θ₀, ε_θ₀, "  "))
        println()
    end
    samples = hmc(target, θ₀, ε_θ₀, n, verbose=verbose)
    if verbose
        println("\nPost θ:")
        samples_θᵢ = [[sample[i] for sample in samples] for i = 1:length(θ₀)]
        θ = mean.(samples_θᵢ)
        ε_θ = quantile.(samples_θᵢ, 0.975) .- quantile.(samples_θᵢ, 0.025)
        print(display(likelihood, θ, ε_θ, "  "))
        println()
        _show_plots(likelihood, θ, x, features=features)
    end
    samples = functional.(samples[1:Int(n / m):end])
    _return_samples && return samples
    ci = Interval{Closed, Closed}(quantile(samples, 0.025), quantile(samples, 0.975))
    return mean(samples), ci
end

function estimate_laplace(
    x,
    prior, α,
    likelihood, θ₀, ε_θ₀,
    functional, ∇functional=nothing;
    features=nothing,
    verbose=true,
    sample=true,
    _return_samples=false,
    _return_mean_std=false
)
    target = _make_target(x, prior, α, likelihood, features=features)
    θ = lbfgs(target, θ₀, verbose=verbose)
    H = ForwardDiff.hessian(target, θ)
    # It can be that the optimisation did not fully converge. Hence, be more careful with
    # the inversion here.
    F = eigen(Symmetric(H))
    any(F.values .>= 0) && @warn "Optimisation failed to properly converge."
    Σ = Symmetric(F.vectors * Diagonal([
        λ >= 0 ? 1e-10 : 1e-10 - inv(λ) for λ in F.values
    ]) * F.vectors')
    ε_θ = 1.96sqrt.(diag(Σ))

    if verbose
        println("\nθ₀:")
        print(display(likelihood, θ₀, ε_θ₀, "  "))
        println("\nMAP θ:")
        print(display(likelihood, θ, ε_θ, "  "))
        println()
        _show_plots(likelihood, θ, x, features=features)
    end

    if sample || _return_samples
        # Propagate uncertainty through sampling.
        L = cholesky(Σ).U'
        θ_samples = L * randn(size(Σ, 1), 200) .+ θ
        samples = [functional(θ_samples[:, i]) for i = 1:200]
        _return_samples && return samples
        ci = Interval{Closed, Closed}(quantile(samples, 0.025), quantile(samples, 0.975))
        return mean(samples), ci
    else
        # Use delta method to approximate distribution of functional.
        ∇ = isnothing(∇functional) ? ForwardDiff.gradient(functional, θ) : ∇functional(θ)
        θ = functional(θ)
        σ = sqrt(dot(∇, Σ * ∇))
        _return_mean_std && return θ, σ
        ε = 1.96σ
        return θ, Interval{Closed, Closed}(θ - ε, θ + ε)
    end
end

"""
    estimate_metric(
        metric,
        x,
        [y];
        features=nothing,
        future_feature=nothing,
        components::Integer=1,
        method::Symbol=:hmc,
        laplace_sample::Bool=true,
        verbose::Bool=false
    )

For high-quality estimates, simply call `estimate_metric(metric, x, [y])`. For faster
estimates, `estimate_metric(metric, x, [y], method=:laplace)` is a good compromise.

# Arguments
- `metric`: Metric to estimate. Must be one of `:mean`, `:median`, `:ew`, `:es`,
    `:mean_over_es`, `:median_over_es`, `:ew_over_es`. You can also give the corresponding
    functions from Metrics.jl.
- `x`: Data to estimate `metric` from.
- `y`: Optional second data set. If provided, the function will estimate the _difference_
    of `metric` between `x` and `y`.

# Keywords
- `features`: Optional features to include in the estimation. This is barely tested.
- `future_feature`: If `features` is provided, this sets the feature where the metric is
    estimated at. For example, if `features` are volumes, then we can only estimate `metric`
    at a given volume. In such cases, `future_feature=mean(features)` could makes sense.
- `components::Integer=1`: Number of components in the mixture. Must be a number between
    `1` and `4`. The first component is an asymmetric Student's t, and all other components
    are asymmetric Gaussians. Higher numbers of components give more flexible models. Note
    that these models are significantly more expensive and may be prone to overfitting.
- `method::Symbol=:hmc`: Method to use. Must be `:laplace` or `:hmc`. `:hmc` is
    significantly more expensive, but may yield better results, i.e. better-calibrated
    confidence intervals.
- `laplace_sample::Bool=true`: When using the Laplace approximation, propagate uncertainties
    using sampling instead of the delta method (linearisation). Sampling is more expensive,
    but may yield better results.
- `verbose::Bool=false`: Print and plot some stuff for you to see what is going on.

# Returns
- `Tuple{Float64, Interval}`: An estimate of the metric and associated confidence interval.
"""
function estimate_metric(
    metric::Symbol,
    x;
    features=nothing,
    future_feature=nothing,
    components::Integer=1,
    method::Symbol=:hmc,
    laplace_sample::Bool=true,
    verbose::Bool=false,
    _return_samples=false,
    _return_mean_std=false
)
    if !isnothing(features)
        # Compress features.
        ndims(features) == 2 && size(features, 2) == 1 && (features = features[:, 1])

        # Estimate feature dependencies with linear regression.
        β₀ = (features' * features) \ (features' * x)

        # Create feature component and ensure that `future_feature` is set.
        if size(β₀) == ()
            feature_component = Linear(β=β₀, ε_β=2abs(β₀))
            isnothing(future_feature) && (future_feature = mean(features))
        else
            feature_component = Linear{length(β₀)}(β=β₀, ε_β=2abs.(β₀))
            isnothing(future_feature) && (future_feature = mean(features, dims=1))
        end
    else
        feature_component = identity
    end

    # Estimate location and width.
    μ = median(x)
    σ = median(abs.(x .- μ))

    # Determine the component locations
    if components == 1
        component_locs = []
        mixture_kw_args = Dict()
    elseif components == 2
        component_locs = [0.5]
        mixture_kw_args = Dict(:π => [1.0, 5.0])
    elseif components == 3
        component_locs = [0.4, 0.6]
        mixture_kw_args = Dict(:π => [1.0, 1.0, 5.0])
    elseif components == 4
        component_locs = [0.3, 0.5, 0.7]
        mixture_kw_args = Dict(:π => [1.0, 1.0, 1.0, 5.0])
    else
        error("Bad number of components $(components). Use a number between 1 and 4.")
    end

    # Define the mixture model.
    likelihood, θ₀, ε_θ₀ = Mixture(
        [(
            Gaussian()
                |> Scale(σ=σ / 2, ε_σ=σ / 2)
                |> Asymmetric()
                |> Location(μ=quantile(x, loc), ε_μ=σ)
        ) for loc in component_locs]...,
        # Be vague about the guessed for the tail parameters. These could be way off!
        StudentsT(ν=2.0, ε_ν=6.0)
            |> Scale(σ=σ, ε_σ=σ)
            |> Asymmetric()
            |> Location(μ=μ, ε_μ=10σ / sqrt(length(x)));
        mixture_kw_args...
    ) |> feature_component

    # Use a log-Gaussian prior centred around the chosen parameters.
    prior, α, _ = Gaussian() |> Scale{length(θ₀)}(σ=ε_θ₀ ./ 1.96) |> Location{length(θ₀)}(μ=θ₀)

    # We need a better tolerance if we are using the Laplace approximation and not sampling.
    if method == :laplace && !laplace_sample
        rtol = 1e-6
    else
        rtol = 1e-3
    end

    # Functionals for the mean:
    functional_mean(θ) = 
        expectation(identity, likelihood, θ, features=future_feature, rtol=rtol)
    ∇functional_mean(θ) = 
        ∇expectation(identity, likelihood, θ, features=future_feature, rtol=rtol)

    # Functionals for the median:
    x₀_median = median(x)
    functional_median(θ) = 
        icdf(likelihood, 0.5, θ, features=future_feature, x₀=x₀_median, rtol=rtol)
    ∇functional_median(θ) = 
        ∇icdf(likelihood, 0.5, θ, features=future_feature, x₀=x₀_median, rtol=rtol)

    # Functionals for ES:
    x₀_es = quantile(x, 0.05)
    functional_es(θ) = 
        es(likelihood, 0.05, θ, features=future_feature, x₀=x₀_es, rtol=rtol)
    ∇functional_es(θ) = 
        ∇es(likelihood, 0.05, θ, features=future_feature, x₀=x₀_es, rtol=rtol)

    # Functionals for EW:
    x₀_ew = quantile(x, 0.95)
    functional_ew(θ) = 
        es(likelihood, 0.95, θ, features=future_feature, x₀=x₀_ew, rtol=rtol)
    ∇functional_ew(θ) = 
        ∇es(likelihood, 0.95, θ, features=future_feature, x₀=x₀_ew, rtol=rtol)

    # Setup call to estimation procedure.
    function _estimate(functional, ∇functional)
        if method == :laplace
            return estimate_laplace(
                x,
                prior, α,
                likelihood, θ₀, ε_θ₀,
                functional, ∇functional,
                verbose=verbose,
                features=features,
                sample=laplace_sample,
                _return_samples=_return_samples,
                _return_mean_std=_return_mean_std
            )
        elseif method == :hmc
            return estimate_hmc(
                x,
                prior, α,
                likelihood, θ₀, ε_θ₀,
                5000, 200,
                functional,
                verbose=verbose,
                features=features,
                _return_samples=_return_samples
            )
        else
            error("Bad method $(method).")
        end
    end

    _make_ratio(f, ∇f, g, ∇g) =  (
        θ -> f(θ) / g(θ),
        θ -> (∇f(θ) .* g(θ) .- f(θ) .* ∇g(θ)) ./ g(θ).^2 
    )

    if metric == :mean
        return _estimate(functional_mean, ∇functional_mean)
    elseif metric == :median
        return _estimate(functional_median, ∇functional_median)
    elseif metric == :ew
        return _estimate(functional_ew, ∇functional_ew)
    elseif metric == :es
        return _estimate(functional_es, ∇functional_es)
    elseif metric == :mean_over_es
        return _estimate(_make_ratio(
            functional_mean, ∇functional_mean,
            functional_es, ∇functional_es
        )...)
    elseif metric == :median_over_es
        return _estimate(_make_ratio(
            functional_median, ∇functional_median,
            functional_es, ∇functional_es
        )...)
    elseif metric == :ew_over_es
        return _estimate(_make_ratio(
            functional_ew, ∇functional_ew,
            functional_es, ∇functional_es
        )...)
    else
        error("Bad metric $(metric).")
    end
end

function estimate_metric(metric, x, y; kw_args...)
    if metric == :mean
        # Exploit linearity of the mean!
        return estimate_metric(:mean, x .- y; kw_args...)
    end
    if _kw(kw_args, :method, :laplace) && _kw(kw_args, :laplace_sample, false)
        μ_x, σ_x = estimate_metric(metric, x; kw_args..., _return_mean_std=true)
        μ_y, σ_y = estimate_metric(metric, y; kw_args..., _return_mean_std=true)
        ε = 1.96sqrt(σ_x^2 + σ_y^2)
        diff = μ_x - μ_y
        return diff, Interval{Closed, Closed}(diff - ε, diff + ε)
    else
        samples_x = estimate_metric(metric, x; kw_args..., _return_samples=true)
        samples_y = estimate_metric(metric, y; kw_args..., _return_samples=true)
        samples = samples_x .- samples_y
        ci = Interval{Closed, Closed}(quantile(samples, 0.025), quantile(samples, 0.975))
        return mean(samples), ci
    end
end

_kw(kw_args, name, value) = name in keys(kw_args) && kw_args[name] == value

estimate_metric(::typeof(Metrics.mean), args...; kw_args...) =
    estimate_metric(:mean, args...; kw_args...)
    
estimate_metric(::typeof(Metrics.median), args...; kw_args...) =
    estimate_metric(:median, args...; kw_args...)
    
estimate_metric(::typeof(Metrics.ew), args...; kw_args...) =
    estimate_metric(:ew, args...; kw_args...)

estimate_metric(::typeof(Metrics.es), args...; kw_args...) =
    estimate_metric(:es, args...; kw_args...)

estimate_metric(::typeof(Metrics.mean_over_es), args...; kw_args...) =
    estimate_metric(:mean_over_es, args...; kw_args...)

estimate_metric(::typeof(Metrics.median_over_es), args...; kw_args...) =
    estimate_metric(:median_over_es, args...; kw_args...)

estimate_metric(::typeof(Metrics.ew_over_es), args...; kw_args...) =
    estimate_metric(:ew_over_es, args...; kw_args...)
    
    
_residualise(d, θ, x; features::Nothing) = d, θ, x

function _residualise(d::Linear{1}, θ, x; features) 
    return (
        d.d,
        θ[2:end],
        x .- features .* θ[1]
    )
end

function _residualise(d::Linear{n}, θ, x; features) where n
    return (
        d.d,
        θ[n + 1:end],
        x .- features * θ[1:n]
    )
end

function _show_plots(likelihood, θ, x; features=nothing)
    # Look at the residuals.
    likelihood, θ, x = _residualise(likelihood, θ, x, features=features)

    ps = collect(2:length(x) - 1) ./ length(x)
    x = sort(x)[2:end - 1]  # Skip minumum and maximum, which are `-Inf` and `Inf`.
    model_quants = @showprogress "Computing model quantiles: " map(zip(ps, x)) do (p, xᵢ)
        icdf(likelihood, p, θ, x₀=xᵢ)
    end

    # Make Q-Q plots.
    function plot_inds(inds)
        err = (
            2sqrt.(ps .* (1 .- ps))
            ./ pdf(likelihood, model_quants, θ)
            ./ sqrt(length(x))
        )
        plot(
            sort(x)[inds],
            (sort(x) .- model_quants)[inds],
            ls="-", lw=0.5, marker="o", markersize=2
        )
        plot(
            sort(x)[inds],
            (sort(x) .- model_quants .- err)[inds],
            c="tab:red", ls="-", lw=0.5, marker="o", markersize=2
        )
        plot(
            sort(x)[inds],
            (sort(x) .- model_quants .+ err)[inds],
            c="tab:red", ls="-", lw=0.5, marker="o", markersize=2
        )
    end

    figure(figsize=(12, 8))

    m = 20

    subplot(2, 2, 1)
    title("All")
    plot_inds(1:length(x))
    ylabel("Data Quantile Overshoot")
    grid()

    subplot(2, 2, 2)
    title("Bulk")
    plot_inds((m + 1):(length(x) - m))
    grid()

    subplot(2, 2, 3)
    title("Left Tail")
    plot_inds(1:m)
    ylabel("Data Quantile Overshoot")
    xlabel("Data Quantile")
    grid()

    subplot(2, 2, 4)
    title("Right Tail")
    plot_inds((length(x) - m + 1):length(x))
    xlabel("Data Quantile")
    grid()

    tight_layout()

    # Plot estimated likelihood.
    figure(figsize=(12, 4))
    x_plot = collect(range(minimum(x), maximum(x), length=1000))
    hist(x, bins=200, density=true)
    plot(x_plot, pdf(likelihood, x_plot, θ))
    scatter(x, x .* 0, s=10, alpha=0.15, edgecolor="none")
    title("Model Likelihood")
    grid()

    show()
end
