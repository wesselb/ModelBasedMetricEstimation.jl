# Parametric Metric Estimation

#### Contents

- [Example: Confidence Intervals For Expected Shortfall](#example-confidence-intervals-for-expected-shortfall)
- [Help!](#help)
- [Ideas for Future Work](#ideas-for-future-work)
- [Coverage Analysis](#coverage-analysis)
- [Power Analysis](#power-analysis)

#### Quick start

```julia
(environment) pkg> add https://gitlab.invenia.ca/wessel.bruinsma/TailPickles.jl#master

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

**WARNING:**
Do not try to look at the source.
It is a mess.

**WARNING:**
These estimators are based on generative models, which in turn rely on assumptions.
They may work well or fail catastrophically.
Care is required.

**DISCLAIMER:**
The code is barely tested.
Use at your own risk.

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

## Coverage Analysis

Coverage for `n = 500` observations is tested in `coverage.jl`.
Below are the results for `1000` repetitions.
For the sake of runtime, the CIs are computed using the Laplace approximation plus a linearisation of the functional, both of which are terrible approximations.
CIs computed with HMC should perform a lot better.

```
StudentsT(ν=2.000, ε_ν=0.000) |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.981 [0.9725338758774952 .. 0.9894661241225048]
Coverage of median: 0.933 [0.917495721872972 .. 0.9485042781270281]
Coverage of ES: 0.909 [0.8911648829047354 .. 0.9268351170952647]
Coverage of mean/ES: 0.841 [0.8183238204381884 .. 0.8636761795618115]
Coverage of median/ES: 0.812 [0.7877712914103967 .. 0.8362287085896034]

StudentsT(ν=5.000, ε_ν=0.000) |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.95 [0.9364848729692 .. 0.9635151270307999]
Coverage of median: 0.947 [0.9331073129901968 .. 0.9608926870098031]
Coverage of ES: 0.957 [0.9444204958155925 .. 0.9695795041844074]
Coverage of mean/ES: 0.952 [0.9387440022980134 .. 0.9652559977019866]
Coverage of median/ES: 0.934 [0.918603616040153 .. 0.9493963839598472]

StudentsT(ν=10.000, ε_ν=0.000) |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.945 [0.9308625664625071 .. 0.9591374335374928]
Coverage of median: 0.924 [0.907567028237079 .. 0.9404329717629211]
Coverage of ES: 0.981 [0.9725338758774952 .. 0.9894661241225048]
Coverage of mean/ES: 0.973 [0.9629489454634204 .. 0.9830510545365796]
Coverage of median/ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]

StudentsT(ν=20.000, ε_ν=0.000) |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.95 [0.9364848729692 .. 0.9635151270307999]
Coverage of median: 0.931 [0.9152828892613825 .. 0.9467171107386176]
Coverage of ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]
Coverage of mean/ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]
Coverage of median/ES: 0.994 [0.9892110296893135 .. 0.9987889703106865]

Gaussian() |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.939 [0.9241587308605473 .. 0.9538412691394526]
Coverage of median: 0.933 [0.917495721872972 .. 0.9485042781270281]
Coverage of ES: 0.969 [0.9582522869532287 .. 0.9797477130467712]
Coverage of mean/ES: 0.991 [0.9851435908674404 .. 0.9968564091325596]
Coverage of median/ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]

Laplace() |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.955 [0.9421447280068441 .. 0.9678552719931558]
Coverage of median: 0.934 [0.9186036160401528 .. 0.9493963839598473]
Coverage of ES: 0.994 [0.9892110296893135 .. 0.9987889703106865]
Coverage of mean/ES: 0.985 [0.9774623308339742 .. 0.9925376691660258]
Coverage of median/ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]

Asymmetric(
    Laplace() |> Scale(σ=1.000, ε_σ=0.000),
    Laplace() |> Scale(σ=2.000, ε_σ=0.000)
) |> Location(μ=5.000, ε_μ=0.000)
Coverage of mean: 0.951 [0.9376136669579637 .. 0.9643863330420362]
Coverage of median: 0.661 [0.6316455528076244 .. 0.6903544471923757]
Coverage of ES: 0.975 [0.965318433353391 .. 0.984681566646609]
Coverage of mean/ES: 0.964 [0.9524478426803156 .. 0.9755521573196844]
Coverage of median/ES: 0.966 [0.954761681166197 .. 0.977238318833803]

Asymmetric(
    Laplace() |> Scale(σ=4.000, ε_σ=0.000),
    Laplace() |> Scale(σ=1.000, ε_σ=0.000)
) |> Location(μ=5.000, ε_μ=0.000)
Coverage of mean: 0.94 [0.925273047731349 .. 0.9547269522686509]
Coverage of median: 0.644 [0.614307876387759 .. 0.6736921236122411]
Coverage of ES: 0.998 [0.9952295290817067 .. 1.0007704709182934]
Coverage of mean/ES: 0.849 [0.8267967961622503 .. 0.8712032038377496]
Coverage of median/ES: 0.633 [0.6031112182102235 .. 0.6628887817897765]
```

Observe that the estimator falls over for the heavy-tailed Student's t.
This can be resolved with better uncertainty, e.g. using `laplace_sample = true`:

```
StudentsT(ν=2.000, ε_ν=0.000) |> Location(μ=5.000, ε_μ=0.000) |> Scale(σ=1.000, ε_σ=0.000)
Coverage of mean: 0.949 [0.9353575766773599 .. 0.96264242332264]
Coverage of median: 0.928 [0.9119707366706115 .. 0.9440292633293886]
Coverage of ES: 0.952 [0.9387440022980134 .. 0.9652559977019866]
Coverage of mean/ES: 0.989 [0.9825320326994487 .. 0.9954679673005513]
Coverage of median/ES: 0.985 [0.9774623308339742 .. 0.9925376691660258]
```

## Power Analysis

Similar.
See `power.jl`.
Below are the results for `1000` repetitions.
Again, for the sake of runtime, the CIs are computed using the Laplace approximation plus a linearisation of the functional, both of which are terrible approximations.

```
Dist. 1:
  Asymmetric(
      StudentsT(ν=1.500, ε_ν=0.000),
      Gaussian() |> Scale(σ=1.000, ε_σ=0.000)
  ) |> Location(μ=0.735, ε_μ=0.000) |> Scale(σ=2650.336, ε_σ=0.000)
Dist. 2:
  Asymmetric(
      Laplace() |> Scale(σ=1.000, ε_σ=0.000),
      Gaussian() |> Scale(σ=2.000, ε_σ=0.000)
  ) |> Location(μ=-0.856, ε_μ=0.000) |> Scale(σ=10969.264, ε_σ=0.000) |> Location(μ=-547.945, ε_μ=0.000)
Desired difference in mean:               547.945205479452
Desired difference in ES                  10000.0
True difference in mean:                  547.9687617117918
True difference in ES:                    -9999.969532725077
True difference in mean/ES:               0.013699157168662997

n: 364
Naive:
  Coverage for difference in mean:          0.943 [0.9286230553267344 .. 0.9573769446732655]
  Power for detecting a difference in mean: 0.128 [0.10728253782822715 .. 0.14871746217177287]
Subsampling:
  Coverage for difference in mean:          0.913 [0.895522941403738 .. 0.9304770585962621]
  Power for detecting a difference in mean: 0.168 [0.1448159325703495 .. 0.19118406742965052]
Parametric:
  Coverage for difference in mean:          0.944 [0.9297421730958736 .. 0.9582578269041263]
  Power for detecting a difference in mean: 0.113 [0.09336755638127434 .. 0.13263244361872567]
Subsampling:
  Coverage for difference in ES:          0.609 [0.5787399004624237 .. 0.6392600995375762]
  Power for detecting a difference in ES: 0.686 [0.6572193757758674 .. 0.7147806242241327]
Parametric:
  Coverage for difference in ES:          0.904 [0.8857319093078249 .. 0.9222680906921752]
  Power for detecting a difference in ES: 0.307 [0.27839716161000866 .. 0.33560283838999133]
Subsampling:
  Coverage for difference in mean/ES:          0.891 [0.8716747363222073 .. 0.9103252636777928]
  Power for detecting a difference in mean/ES: 0.124 [0.10356210221020602 .. 0.144437897789794]
Parametric:
  Coverage for difference in mean/ES:          0.958 [0.9455611357856537 .. 0.9704388642143462]
  Power for detecting a difference in mean/ES: 0.065 [0.0497125231988092 .. 0.0802874768011908]

n: 546
Naive:
  Coverage for difference in mean:          0.952 [0.9387440022980134 .. 0.9652559977019866]
  Power for detecting a difference in mean: 0.135 [0.11380915547006464 .. 0.15619084452993537]
Subsampling:
  Coverage for difference in mean:          0.928 [0.9119707366706115 .. 0.9440292633293886]
  Power for detecting a difference in mean: 0.167 [0.1438711487236966 .. 0.1901288512763034]
Parametric:
  Coverage for difference in mean:          0.954 [0.9410094831539338 .. 0.9669905168460661]
  Power for detecting a difference in mean: 0.129 [0.10821369670179894 .. 0.14978630329820108]
Subsampling:
  Coverage for difference in ES:          0.668 [0.6387967894040893 .. 0.6972032105959107]
  Power for detecting a difference in ES: 0.683 [0.6541455161397949 .. 0.7118544838602052]
Parametric:
  Coverage for difference in ES:          0.924 [0.907567028237079 .. 0.9404329717629211]
  Power for detecting a difference in ES: 0.272 [0.24440544029477324 .. 0.2995945597052268]
Subsampling:
  Coverage for difference in mean/ES:          0.918 [0.9009861893935736 .. 0.9350138106064265]
  Power for detecting a difference in mean/ES: 0.094 [0.07590321807845617 .. 0.11209678192154383]
Parametric:
  Coverage for difference in mean/ES:          0.961 [0.9489948573582048 .. 0.9730051426417952]
  Power for detecting a difference in mean/ES: 0.056 [0.0417421730958737 .. 0.07025782690412631]

n: 728
Naive:
  Coverage for difference in mean:          0.936 [0.920822465155238 .. 0.9511775348447621]
  Power for detecting a difference in mean: 0.18 [0.15617589985439642 .. 0.20382410014560357]
Subsampling:
  Coverage for difference in mean:          0.904 [0.8857319093078249 .. 0.9222680906921752]
  Power for detecting a difference in mean: 0.227 [0.2010237770690626 .. 0.2529762229309374]
Parametric:
  Coverage for difference in mean:          0.933 [0.917495721872972 .. 0.9485042781270281]
  Power for detecting a difference in mean: 0.174 [0.15049079326729203 .. 0.19750920673270794]
Subsampling:
  Coverage for difference in ES:          0.644 [0.614307876387759 .. 0.6736921236122411]
  Power for detecting a difference in ES: 0.719 [0.6911265367714675 .. 0.7468734632285324]
Parametric:
  Coverage for difference in ES:          0.909 [0.8911648829047354 .. 0.9268351170952647]
  Power for detecting a difference in ES: 0.327 [0.29790926221931413 .. 0.3560907377806859]
Subsampling:
  Coverage for difference in mean/ES:          0.92 [0.9031766595236028 .. 0.9368233404763973]
  Power for detecting a difference in mean/ES: 0.093 [0.07498980375540652 .. 0.11101019624459348]
Parametric:
  Coverage for difference in mean/ES:          0.938 [0.9230455449672906 .. 0.9529544550327093]
  Power for detecting a difference in mean/ES: 0.085 [0.06770608432079306 .. 0.10229391567920695]

n: 910
Naive:
  Coverage for difference in mean:          0.946 [0.9319842682982609 .. 0.960015731701739]
  Power for detecting a difference in mean: 0.198 [0.17328884148901647 .. 0.22271115851098355]
Subsampling:
  Coverage for difference in mean:          0.906 [0.8879032180784562 .. 0.9240967819215439]
  Power for detecting a difference in mean: 0.252 [0.22507694139432363 .. 0.2789230586056764]
Parametric:
  Coverage for difference in mean:          0.94 [0.925273047731349 .. 0.9547269522686509]
  Power for detecting a difference in mean: 0.219 [0.19335392537748625 .. 0.24464607462251375]
Subsampling:
  Coverage for difference in ES:          0.63 [0.6000605054596219 .. 0.6599394945403781]
  Power for detecting a difference in ES: 0.732 [0.7045339463026501 .. 0.7594660536973499]
Parametric:
  Coverage for difference in ES:          0.911 [0.8933425689504031 .. 0.928657431049597]
  Power for detecting a difference in ES: 0.343 [0.3135623544321803 .. 0.3724376455678198]
Subsampling:
  Coverage for difference in mean/ES:          0.929 [0.9130738662527237 .. 0.9449261337472764]
  Power for detecting a difference in mean/ES: 0.085 [0.06770608432079306 .. 0.10229391567920695]
Parametric:
  Coverage for difference in mean/ES:          0.954 [0.9410094831539338 .. 0.9669905168460661]
  Power for detecting a difference in mean/ES: 0.095 [0.07681725619791743 .. 0.11318274380208257]

n: 1092
Naive:
  Coverage for difference in mean:          0.932 [0.9163888196938684 .. 0.9476111803061317]
  Power for detecting a difference in mean: 0.238 [0.2115917701425445 .. 0.2644082298574555]
Subsampling:
  Coverage for difference in mean:          0.9 [0.881396503283251 .. 0.918603496716749]
  Power for detecting a difference in mean: 0.293 [0.26477611172305504 .. 0.3212238882769449]
Parametric:
  Coverage for difference in mean:          0.927 [0.9108685047581219 .. 0.9431314952418782]
  Power for detecting a difference in mean: 0.25 [0.22314816540679164 .. 0.27685183459320833]
Subsampling:
  Coverage for difference in ES:          0.63 [0.6000605054596219 .. 0.6599394945403781]
  Power for detecting a difference in ES: 0.787 [0.761610713569195 .. 0.8123892864308051]
Parametric:
  Coverage for difference in ES:          0.925 [0.9086666666666667 .. 0.9413333333333334]
  Power for detecting a difference in ES: 0.373 [0.34301103249907106 .. 0.40298896750092894]
Subsampling:
  Coverage for difference in mean/ES:          0.918 [0.9009861893935736 .. 0.9350138106064265]
  Power for detecting a difference in mean/ES: 0.101 [0.08231410689849156 .. 0.11968589310150846]
Parametric:
  Coverage for difference in mean/ES:          0.945 [0.9308625664625071 .. 0.9591374335374928]
  Power for detecting a difference in mean/ES: 0.096 [0.07773190930782482 .. 0.11426809069217518]
```
