# Parametric Metric Estimation

#### Contents

- [Important Notes](#important-notes)
- [Quick Start](#quick-start)
- [Example: Confidence Intervals For Expected Shortfall](#example-confidence-intervals-for-expected-shortfall)
- [Help!](#help)
- [Ideas for Future Work](#ideas-for-future-work)

## Important Notes

These estimators are based on a generative models, which in turn rely
on assumptions.
They may work well or fail catastrophically.
Care is required.
Moreover, this implementation is ported over from quick and dirty squad work.
As a result, the code is a little messy and big parts of the code are not
properly tested.
Use at your own risk.

## Quick start

```julia
julia> using TailPickles, Statistics

julia> estimate_metric(mean, randn(100))
(0.024958916510256165, Intervals.Interval{Float64,Intervals.Closed,Intervals.Closed}(-0.20370101694421278, 0.2945224118131819))
```

This package exports just one function for you to use:

```
help?> estimate_metric
search: estimate_metric

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

  For high-quality estimates, simply call estimate_metric(metric, x, [y]). For faster estimates, estimate_metric(metric, x, [y],
  method=:laplace) is a good compromise.

  Arguments
  ≡≡≡≡≡≡≡≡≡≡≡

    •    metric: Metric to estimate. Must be one of :mean, :median, :ew, :es, :mean_over_es, :median_over_es, :ew_over_es. You
        can also give the corresponding functions from Metrics.jl.

    •    x: Data to estimate metric from.

    •    y: Optional second data set. If provided, the function will estimate the difference of metric between x and y.

  Keywords
  ≡≡≡≡≡≡≡≡≡≡

    •    features: Optional features to include in the estimation. This is barely tested.

    •    future_feature: If features is provided, this sets the feature where the metric is estimated at. For example, if
        features are volumes, then we can only estimate metric at a given volume. In such cases, future_feature=mean(features)
        could makes sense.

    •    components::Integer=1: Number of components in the mixture. Must be a number between 1 and 4. The first component is an
        asymmetric Student's t, and all other components are asymmetric Gaussians. Higher numbers of components give more
        flexible models. Note that these models are significantly more expensive and may be prone to overfitting.

    •    method::Symbol=:hmc: Method to use. Must be :laplace or :hmc. :hmc is significantly more expensive, but may yield
        better results, i.e. better-calibrated confidence intervals.

    •    laplace_sample::Bool=true: When using the Laplace approximation, propagate uncertainties using sampling instead of the
        delta method (linearisation). Sampling is more expensive, but may yield better results.

    •    verbose::Bool=false: Print and plot some stuff for you to see what is going on.

  Returns
  ≡≡≡≡≡≡≡≡≡

    •    Tuple{Float64, Interval}: An estimate of the metric and associated confidence interval.

```

## Example: Confidence Intervals For Expected Shortfall

```julia
julia> using TailPickles, Metrics, Distributions, ProgressMeter

julia> x = rand(Distributions.TDist(2.5), 500);

julia> es(x)
3.9298862017846146

julia> @time estimate_metric(es, x)
 18.909742 seconds (22.24 M allocations: 30.620 GiB, 28.05% gc time)
(4.238016065638495, Intervals.Interval{Float64,Intervals.Closed,Intervals.Closed}(3.3429874113836213, 6.217255521253619))

julia> @time estimate_metric(es, x, components=2)  # Pushing the method to its limits.
415.197915 seconds (108.54 M allocations: 423.125 GiB, 17.98% gc time)
(4.49663954914686, Intervals.Interval{Float64,Intervals.Closed,Intervals.Closed}(3.3187778461836412, 7.600391091124792))

julia> @time estimate_metric(es, x, method=:laplace)  # This is a cheap approximation.
  0.322232 seconds (5.17 M allocations: 107.069 MiB, 14.28% gc time)
(4.159180536725604, Intervals.Interval{Float64,Intervals.Closed,Intervals.Closed}(3.273044330090013, 5.255048956899458))

julia> truth = es(rand(TDist(2.5), 10_000_000))  # Let's see how well we did.
4.598287468990342

julia> mean(@showprogress [truth in estimate_metric(es, rand(TDist(2.5), 500), method=:laplace)[2] for i = 1:200])
Progress: 100%|███████████████████████████████████████████████████████████████████████████████| Time: 0:01:04
0.95

julia> truth = es(rand(Laplace(), 10_000_000))  # Let's also check the case of a Laplace distribution.
3.3030984650297537

julia> mean(@showprogress [truth in estimate_metric(es, rand(Laplace(), 500), method=:laplace)[2] for i = 1:200])
Progress: 100%|███████████████████████████████████████████████████████████████████████████████| Time: 0:01:10
0.95
```

## Help!

If the results are look like garbage, try setting `verbose=true` to see what is going on.
You will be shown a lot more detail and also QQ-plots to assess the model fit.

## Ideas for Future Work

- Model dependence on features, e.g. temperature or sum of absolute MCCs.

- Model autocorrelations in the returns or in volatility of the returns.

-
    Induce more powerful tests by exploiting correlations between two time series.
    For example, impose a positively correlated joint Gaussian prior over the two tail indices.
    This is already exploited in testing for differences between their means, where we look at the mean of the differences instead.
    In that case, if the two time series are positively correlated, we get some variance reduction.
