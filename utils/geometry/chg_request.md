# PINN Geometry Request

**Shape Name:** Cylinder Head Gasket

**Description:**
Use reference image mentioned in Geometry Configuration. It is a plate with 4 big circular holes, named cylinder holes, several smaller holes for screws and some random shaped holes, smaller than main holes. You are free too choose the random holes sizes.
The plate shape has height and width, available below. Mind that screw holes dont intersect. distribute randomly on plate.
Mind a space between cylinder holes (space).

**Geometry Configuration:**
- Number of cylinder holes: 4
- Holes Radius: 0.1 m
- Screw holes: 0.01 m at most
- Plate heigh: .25 m
- Plate Width: 1 m
- Number of screw holes: 10
- Space: 0.005 m
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