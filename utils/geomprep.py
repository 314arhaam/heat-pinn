import shapely
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely.geometry as geometry
from shapely import Polygon, Point, MultiPoint
from matplotlib.path import Path
from shapely.geometry import GeometryCollection

class Geometry:
    def __init__(self, geometry: shapely.geometry) -> None:
        self.geometry = geometry
    

class Geometry2D(Geometry):
    def __init__(self, geometry: shapely.geometry) -> None:
        super().__init__(geometry)
    
    def makeBoundary(self, n: int) -> None:
        geometry = self.geometry
        n += 1
        N, data = [], []
        total_perim = geometry.length
        distances = np.linspace(0, total_perim, n)
        for geom in list(geometry.boundary.geoms):
            perim = geom.length
            N.append(int(n*perim/total_perim))
        for n, geom in zip(N, list(geometry.boundary.geoms)):
            distances = np.linspace(0, geom.length, 5*n)
            distances = distances[list(np.random.randint(0, 5*n, n))]
            for d in distances:
                data.append(*geom.interpolate(d).coords)
        self.boundaryDataFrame = pd.DataFrame(data, columns=["x", "y"])
        self.boundaryDataFrame['value'] = 0.
        return None
    
    def makeDomain(self, n: int, tolcoef: int = 4) -> None:
        geometry = self.geometry
        xmin, ymin, xmax, ymax = geometry.bounds
        random_domain = np.random.rand(tolcoef*n, 2)
        random_domain[:, 0] = random_domain[:, 0] * (xmax - xmin) + xmin
        random_domain[:, 1] = random_domain[:, 1] * (ymax - ymin) + ymin
        random_domain = MultiPoint(random_domain)
        domain = shapely.intersection(geometry, random_domain)
        self.domain = np.array(list(map(lambda p: [p.x, p.y], list(domain.geoms))))
        return None
    
    def plot(self):
        xb, yb = self.boundaryDataFrame['x'], self.boundaryDataFrame['y']
        xd, yd = self.domain[:, 0], self.domain[:, 1]
        plt.scatter(xb, yb, c='k', marker='x')
        plt.scatter(xd, yd, c='r', marker='.')
        return None
