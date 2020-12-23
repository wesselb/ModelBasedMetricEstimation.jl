@testset "Estimate" begin
    for metric in [mean, median, ew, es, mean_over_es, median_over_es, ew_over_es]
        # Just run the estimator to check that everything runs.
        # HMC:
        estimate_metric(metric, randn(50), method=:hmc)
        estimate_metric(metric, randn(50), randn(50), method=:hmc)
        # Laplace with sampling:
        estimate_metric(metric, randn(50), method=:laplace, laplace_sample=true)
        estimate_metric(metric, randn(50), randn(50), method=:laplace, laplace_sample=true)
        # Laplace without sampling:
        estimate_metric(metric, randn(50), method=:laplace, laplace_sample=false)
        estimate_metric(metric, randn(50), randn(50), method=:laplace, laplace_sample=false)
    end

    for components in [1, 2, 3, 4]
        # Just run the cheapest estimator to check that it runs for every number of
        # components.
        estimate_metric(
            mean,
            randn(50),
            components=components,
            method=:laplace,
            laplace_sample=false
        )
    end
end
