# Verify GenJAX can define and run a simple generative function.
import jax
import jax.numpy as jnp
import genjax

# Simple linear regression model
@genjax.gen
def model(x):
    slope = genjax.normal(0.0, 10.0) @ "slope"
    intercept = genjax.normal(0.0, 10.0) @ "intercept"
    y = genjax.normal(slope * x + intercept, 1.0) @ "y"
    return y

# Simulate
tr = model.simulate(2.0)
print("=== GenJAX Test ===")
print(f"Trace choices: slope={tr['slope']:.4f}, intercept={tr['intercept']:.4f}, y={tr['y']:.4f}")
print(f"Score: {tr.get_score():.4f}")
print(f"Retval: {tr.get_retval():.4f}")

# Generate with constraint
constraints = {"y": 5.0}
tr2, weight = model.generate(constraints, 2.0)
print(f"\nGenerate with y=5.0 constraint:")
print(f"  slope={tr2['slope']:.4f}, intercept={tr2['intercept']:.4f}")
print(f"  weight={weight:.4f}")
print("\nGenJAX: ALL OK")
