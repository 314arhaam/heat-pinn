from geomprep import Geometry2D
from shapely import Polygon, Point, MultiPoint

if __name__ == "__main__":
    # create a polygon
    p1 = Polygon([[-1, -1], [-1, +1], [+1, +1], [+1, -1], [-1, -1]])
    # create a circle
    c1 = Point(0, -0.5).buffer(0.3)
    # bore the circle inside the polygon
    geom = p1.difference(c1)
    # show the *.svg
    geom
    # create Geometry2D object
    G = Geometry2D(geom)
    # make boundary; 100 points
    G.makeBoundary(100)
    # make domain; 500 points
    G.makeDomain(500)
    # show scatter plot
    G.plot()