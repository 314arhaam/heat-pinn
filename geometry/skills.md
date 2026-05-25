# AI Capabilities: PINN Data Generation

This module allows the AI to perform the following "Skills":

1. **Complex Manifold Generation:** Creating gears, airfoils, stars, and multi-hole domains using boolean geometry.
2. **Type-Safe Data Export:** Automatically casting all arrays to `float64` to ensure compatibility with high-precision TensorFlow models and custom `@tf.function` wrappers.
3. **Equidistant Boundary Sampling:** Calculating the perimeter length and placing points at exact intervals to ensure the PINN doesn't "ignore" parts of the boundary.
4. **Collision Detection:** Ensuring domain points are strictly "inside" the shape and strictly "outside" any holes using `shapely.contains`.