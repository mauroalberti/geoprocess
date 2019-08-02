

from osgeo import ogr

multiline1 = ogr.Geometry(ogr.wkbMultiLineString)

line1 = ogr.Geometry(ogr.wkbLineString)
line1.AddPoint(1214242.4174581182, 617041.9717021306)
line1.AddPoint(1234593.142744733, 629529.9167643716)
multiline1.AddGeometry(line1)

line2 = ogr.Geometry(ogr.wkbLineString)
line2.AddPoint(1184641.3624957693, 626754.8178616514)
line2.AddPoint(1219792.6152635587, 606866.6090588232)
multiline1.AddGeometry(line2)

line3 = ogr.Geometry(ogr.wkbLineString)
line3.AddPoint(1184641.3624957693, 626754.8178616514)
line3.AddPoint(1219792.6152635587, 606866.6090588232)
multiline1.AddGeometry(line3)

print ("Geometry has %i geometries" % (multiline1.GetGeometryCount()))

for ndx, geom in enumerate(multiline1):

    print (ndx, type(geom), geom)

