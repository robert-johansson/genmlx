# Verify Gen.jl can define and run a simple generative function.
using Gen

# Simple linear regression model
@gen function model(x::Float64)
    slope = @trace(normal(0.0, 10.0), :slope)
    intercept = @trace(normal(0.0, 10.0), :intercept)
    y = @trace(normal(slope * x + intercept, 1.0), :y)
    return y
end

# Simulate
tr = simulate(model, (2.0,))
println("=== Gen.jl Test ===")
println("Trace choices: slope=$(round(tr[:slope], digits=4)), intercept=$(round(tr[:intercept], digits=4)), y=$(round(tr[:y], digits=4))")
println("Score: $(round(get_score(tr), digits=4))")
println("Retval: $(round(get_retval(tr), digits=4))")

# Generate with constraint
constraints = choicemap((:y, 5.0))
tr2, weight = generate(model, (2.0,), constraints)
println("\nGenerate with y=5.0 constraint:")
println("  slope=$(round(tr2[:slope], digits=4)), intercept=$(round(tr2[:intercept], digits=4))")
println("  weight=$(round(weight, digits=4))")
println("\nGen.jl: ALL OK")
