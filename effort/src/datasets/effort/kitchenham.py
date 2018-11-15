"""
# KitchenHam dataset from promise

Standard header:

"""
from __future__ import division,print_function
import  sys
sys.dont_write_bytecode = True
from lib import *

"""
@attribute Client.code {1,2,3,4,5,6}
@attribute Project.type {A,C,D,P,Pr,U}
@attribute Actual.duration numeric
@attribute Adjusted.function.points numeric
@attribute First.estimate numeric
@attribute First.estimate.method {A,C,CAE,D,EO,W}
@attribute Actual.effort numeric
"""

def run(weighFeature = False, 
           split = "median"):
  vl=1;l=2;n=3;h=4;vh=5;xh=6;_=0
  A=1;C=2;D=3;P=CAE=4;Pr=EO=5;U=W=6
  return data(indep= [
     # 0..5
    'code','type','duration','function_points','estimate','estimate_method'],
    less = ['Effort'],
    _rows=[
      [1,A,107,101.65,495,EO,485],
      [1,D,144,57.12,1365,A,990],
      [1,D,604,1010.88,8244,EO,13635],
      [1,P,226,45.6,1595,D,1576],
      [1,D,326,1022.58,3828,A,3826],
      [1,P,294,77.04,879,EO,1079],
      [1,A,212,159.6,2895,EO,2224],
      [1,C,175,225.54,1800,A,1698],
      [1,Pr,584,144.72,1160,EO,1173],
      [1,D,171,84.42,885,EO,1401],
      [1,D,201,126.42,2125,EO,2170],
      [1,D,195,392.16,1381,EO,1122],
      [1,U,109,18.9,1142,D,1024],
      [1,P,263,112.14,1895,EO,1832],
      [1,A,165,210.08,1339,EO,1016],
      [1,D,46,260.95,447,EO,322],
      [2,D,186,609.7,507,EO,580],
      [2,D,189,169.85,952,EO,1003],
      [2,P,95,56,380,EO,380],
      [2,P,53,30,220,EO,220],
      [2,P,365,241.86,2879,EO,2356],
      [2,P,438,219.88,1483,EO,1388],
      [2,P,109,229.71,1667,EO,1066],
      [2,D,283,458.38,2125,A,2860],
      [2,P,137,177.63,1175,A,1143],
      [2,P,102,287.64,2213,A,1431],
      [2,P,103,343.54,2247,A,1868],
      [2,P,192,346.8,1926,A,2172],
      [2,D,219,1121.48,5641,EO,8509],
      [2,P,484,761.08,3928,EO,5927],
      [2,P,173,464,1995,A,2663],
      [2,P,169,203.01,2281,EO,1425],
      [2,P,207,550.14,3305,EO,3504],
      [2,P,61,86.45,797,EO,652],
      [2,P,311,1362.11,3922,A,7649],
      [2,P,418,681,6809,A,5927],
      [2,P,410,485.1,4955,A,6607],
      [2,0,497,172.96,1294,EO,2591],
      [2,P,259,2075.8,5688,EO,4494],
      [2,D,234,756.25,5245,EO,4824],
      [2,0,462,789.66,3930,EO,5094],
      [2,P,291,357,2562,EO,3088],
      [2,P,116,62.08,1526,C,892],
      [2,P,128,157.56,1018,EO,750],
      [2,D,185,322.62,5646,CAE,5646],
      [2,P,207,81.34,1532,EO,1532],
      [2,P,151,191,1532,EO,1280],
      [2,P,99,121.52,314,EO,313],
      [2,P,61,222.78,412,A,339],
      [2,P,101,113.52,738,EO,583],
      [2,0,462,15.36,763,EO,726],
      [2,P,133,320.12,1750,A,1939],
      [2,P,106,84.28,682,A,669],
      [2,P,68,248.88,1320,EO,1413],
      [2,P,239,616.32,3573,EO,4115],
      [2,P,330,515.07,2913,A,4009],
      [2,P,37,88.2,701,EO,630],
      [2,P,187,115.14,725,A,718],
      [2,P,329,63.84,1826,A,1584],
      [2,P,120,1015.98,5000,EO,5816],
      [2,P,85,359.64,2640,A,2037],
      [2,P,49,240.84,2534,A,1428],
      [2,P,152,285.12,2231,A,1252],
      [2,P,47,61.2,1000,EO,655],
      [2,0,148,287.28,1645,D,1318],
      [2,D,141,172,1067,EO,995],
      [2,D,235,144.06,2270,EO,2265],
      [2,D,298,108.64,656,EO,654],
      [2,0,99,165.36,121,A,718],
      [2,P,127,680.9,1685,EO,2029],
      [2,D,163,409.4,2350,EO,1650],
      [2,D,316,313.95,2308,EO,2223],
      [2,D,388,1136.34,7850,A,8600],
      [2,P,152,177,2004,EO,1884],
      [2,D,166,746.24,3715,W,5359],
      [2,P,114,274.92,1273,A,1159],
      [2,P,82,43.5,437,A,437],
      [2,D,123,54.75,813,EO,570],
      [2,P,49,130,900,EO,502],
      [2,P,183,525.96,2475,A,1877],
      [2,P,149,311.85,799,A,1693],
      [2,0,370,1185.08,2160,EO,3319],
      [2,D,128,258.24,1770,EO,1557],
      [2,P,126,60,760,EO,557],
      [2,P,200,303.52,2588,A,3050],
      [2,D,76,98.9,1148,A,1113],
      [2,P,299,711.9,4064,EO,5456],
      [2,D,131,182.4,933,EO,763],
      [2,0,140,351.9,2096,EO,2203],
      [2,P,169,401.98,3284,EO,3483],
      [2,P,130,162.61,4576,A,2393],
      [2,D,389,1210.99,14226,EO,15673],
      [2,D,166,156.42,6080,EO,2972],
      [2,P,148,603.58,4046,EO,4068],
      [2,P,131,73.92,649,EO,698],
      [2,P,144,121.55,817,EO,676],
      [2,0,369,1234.2,6340,EO,6307],
      [2,P,155,35,300,EO,219],
      [2,0,102,61.06,315,EO,254],
      [2,P,149,162,750,EO,324],
      [2,P,548,1285.7,898,EO,874],
      [2,D,946,18137.48,79870,EO,113930],
      [2,D,186,1020.6,1600,EO,1722],
      [2,D,212,377,1702,EO,1660],
      [2,P,84,210.45,592,EO,693],
      [2,D,250,410,2158,EO,1455],
      [2,D,86,279,994,EO,988],
      [2,D,102,240,1875,EO,1940],
      [2,P,137,230,2527,EO,2408],
      [2,D,87,150.29,2606,EO,1737],
      [2,D,291,1940.68,12694,EO,12646],
      [2,D,392,401,4176,EO,4414],
      [2,D,165,267,2240,EO,2480],
      [2,D,88,102,980,EO,980],
      [2,D,249,403,3720,EO,3189],
      [2,D,186,857,2914,EO,2895],
      [2,D,63,69,360,EO,322],
      [2,A,192,980.95,3700,EO,3555],
      [2,P,123,100.8,200,EO,570],
      [2,P,123,105.28,578,EO,464],
      [2,D,186,158.4,1652,EO,1742],
      [2,D,119,219.88,780,A,896],
      [2,P,195,1292.56,8690,EO,8656],
      [2,P,210,616.08,3748,EO,3966],
      [2,D,180,103.4,710,EO,989],
      [2,P,238,74.4,856,EO,585],
      [2,P,144,356.31,2436,EO,1860],
      [2,P,432,862,4101,EO,5249],
      [2,P,392,791.84,5231,EO,5192],
      [2,D,205,661.27,2853,A,1832],
      [2,D,49,179,1246,EO,928],
      [3,P,205,518.4,2570,EO,2570],
      [3,D,145,370,1328,EO,1328],
      [3,D,172,839.05,3380,EO,2964],
      [3,P,137,243.86,1522,EO,1304],
      [4,D,371,557.28,2264,EO,1631],
      [4,C,217,485.94,2790,EO,955],
      [4,D,308,698.54,1312,EO,286],
      [4,D,217,752.64,2210,A,1432],
      [5,D,40,809.25,337,EO,321],
      [6,P,253,178.1,865,EO,593],
      [6,P,405,81.48,441,EO,302],
      [6,P,241,1093.86,2731,EO,2634],
      [6,0,156,1002.76,1039,EO,1040],
      [2,D,92,551.88,1393,A,887]
    ],
    _tunings =[[
    #         vlow  low   nom   high  vhigh xhigh
    #scale factors:
    'Prec',   6.20, 4.96, 3.72, 2.48, 1.24, _ ],[
    'Flex',   5.07, 4.05, 3.04, 2.03, 1.01, _ ],[
    'Resl',   7.07, 5.65, 4.24, 2.83, 1.41, _ ],[
    'Pmat',   7.80, 6.24, 4.68, 3.12, 1.56, _ ],[
    'Team',   5.48, 4.38, 3.29, 2.19, 1.01, _ ]],
    weighFeature = weighFeature,
    _split = split,
    _isCocomo = False
    )

def _kitchenham(): print(kitchenham())
