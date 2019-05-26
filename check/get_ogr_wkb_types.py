# from: https://gist.github.com/CMCDragonkai/e7b15bb6836a7687658ec2bb3abd2927

import pprint
from osgeo import ogr
pprint.pprint(list(map(lambda f: (f, getattr(ogr, f)), list(filter(lambda x: x.startswith('wkb'), dir(ogr))))))

