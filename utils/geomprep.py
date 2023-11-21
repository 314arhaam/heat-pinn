import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely import MultiPoint
# import shapely.geometry as geometry
# from matplotlib.path import Path
# from shapely.geometry import GeometryCollection


"""
WARNING:
Currently it only works fine with multi geometry shapely objects, have bugs when
there are simple shapely geometries.
"""

class Geometry:
    """Geometry object
    The purpose of this class is to convert a shapely.geometry object to a
    point cloud of boundary points and domain points that could be used for
    our PINN algorithm.
    """
    def __init__(self, geometry: shapely.geometry) -> None:
        """Initialize the object

        Args:
            geometry (shapely.geometry): The shapely freindly geometry object
            to convert to set of points.
        """
        self.geometry = geometry
    

class Geometry2D(Geometry):
    """2-dimensional geometry
    """
    def __init__(self, geometry: shapely.geometry) -> None:
        """Initialize the object

        Args:
            geometry (shapely.geometry): The shapely freindly geometry object
            to convert to set of points.
        """
        super().__init__(geometry)
    
    def makeBoundary(self, n: int) -> None:
        """This method is used to generate random points on all of the geometry
        boundaries.

        Args:
            n (int): total number of points on all boundaries.

        Returns:
            None
        """
        geometry = self.geometry
        n += 1
        # initialize data lists
        N, data = [], []
        # measure the total perimeter of the geometry boundary
        total_perim = geometry.length
        # create n evenly distributed points on the whole perimeter
        distances = np.linspace(0, total_perim, n)
        # number of points on each boundary side must be porpotional to its 
        # length. In other words, the percentage of the length it has.
        for geom in list(geometry.boundary.geoms):
            perim = geom.length
            N.append(int(n*perim/total_perim))
        # walk on the perimeter
        for n, geom in zip(N, list(geometry.boundary.geoms)):
            distances = np.linspace(0, geom.length, 5*n)
            distances = distances[list(np.random.randint(0, 5*n, n))]
            for d in distances:
                data.append(*geom.interpolate(d).coords)
        # generate padnas.DataFrame object for boundary points
        self.boundaryDataFrame = pd.DataFrame(data, columns=["x", "y"])
        # set the BC value for each point to default: 0.
        self.boundaryDataFrame['value'] = 0.
        return None
    
    def makeDomain(self, n: int, tolcoef: int = 4) -> None:
        """This method is used to randomly distribute points on the main domain
        of the geometry.

        Args:
            n (int): total number of the points inside the geometry domain.
            tolcoef (int, optional): For cases with a hole or subtracted geometry,
            use this parameter to recover the number of points. Defaults to 4.

        Returns:
            None
        """
        geometry = self.geometry
        # find bounds or envelope corners
        xmin, ymin, xmax, ymax = geometry.bounds
        random_domain = np.random.rand(tolcoef*n, 2)
        # defining the `envelope` as the biggest rectangle that covers all over 
        # the shape, here we distribute the random numbers over the envelope 
        # that covers the shapely.geometry object.
        random_domain[:, 0] = random_domain[:, 0] * (xmax - xmin) + xmin
        random_domain[:, 1] = random_domain[:, 1] * (ymax - ymin) + ymin
        # covert the `numpy.array` random points to `shapely.MultiPoint` object 
        # so they could be easily intersected with the desired geometry
        random_domain = MultiPoint(random_domain)
        domain = shapely.intersection(geometry, random_domain)
        # covert to `numpy.array`, maybe a `pandas.DataFrame` in future
        self.domain = np.array(list(map(lambda p: [p.x, p.y], list(domain.geoms))))
        return None
    
    def plot(self):
        """Plot the boundary and domain points, as an scatter plot. Useful to 
        check if everything is generated correctly, according to the desired
        shapely.geometry.

        Returns:
            None
        """
        # just making the xs & ys easier to call.
        xb, yb = self.boundaryDataFrame['x'], self.boundaryDataFrame['y']
        xd, yd = self.domain[:, 0], self.domain[:, 1]
        # scatter plot
        plt.scatter(xb, yb, c='k', marker='x') # black 'X's for boundary points
        plt.scatter(xd, yd, c='r', marker='.') # red dots for domain points
        return None
