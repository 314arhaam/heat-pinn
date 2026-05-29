# PINN Geometry Request

**Shape Name:** Cylinder Head Gasket

**Description:**
Use reference image mentioned in Geometry Configuration. It is a plate with 4 big circular holes, several smaller holes for screws and some random shaped holes, smaller than main holes. You are free too choose the random holes sizes.

**Geometry Configuration:**
- Number of holes: 4
- Holes Radius: 0.1 m
- Screw holes: 0.01 m at most
- Reference image: https://imagedelivery.net/IaKGQXu3HHhBuEAF7BOmlg/c90aed01-ce7e-408f-ab78-1ebbda067100/public

**Data Requirements:**
- Total Domain Points (Internal): 20k
- Total Boundary Points (Edges): 2k

**Physics Metadata (Boundary Values):**
- Outer Boundary Value (t): 50
- Inner Boundary Value (t): 90

**Technical Specs:**
- Precision: float64
- Export Format: Parquet
- Plotting: Required