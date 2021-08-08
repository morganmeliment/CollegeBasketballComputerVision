import numpy as np

goodStr = [6 * i for i in range(35, 106)]

dictionaryStr = eval("{384: [214.0, 1.0297443], 258: [230.0, 1.012291], 516: [255.0, 1.012291], 390: [207.0, 1.0297443], 264: [250.0, 1.012291], 522: [260.0, 1.012291], 396: [202.0, 1.012291], 270: [265.0, 1.012291], 528: [267.0, 1.012291], 600: [243.0, 0.78539819], 402: [194.0, 1.0297443], 276: [281.0, 1.012291], 534: [275.0, 1.012291], 408: [190.0, 1.0297443], 282: [292.0, 1.012291], 540: [287.0, 1.012291], 414: [188.0, 1.0297443], 288: [294.0, 1.012291], 624: [363.0, 1.012291], 546: [305.0, 1.012291], 420: [194.0, 1.012291], 294: [290.0, 1.012291], 552: [257.0, 1.0646509], 426: [200.0, 1.0297443], 300: [286.0, 1.012291], 558: [353.0, 1.012291], 432: [216.0, 1.0297443], 306: [279.0, 1.012291], 564: [372.0, 1.012291], 606: [334.0, 1.0646509], 438: [234.0, 1.012291], 312: [272.0, 1.012291], 570: [384.0, 1.012291], 444: [249.0, 1.012291], 318: [269.0, 1.012291], 576: [398.0, 1.012291], 450: [262.0, 1.012291], 324: [269.0, 1.012291], 582: [403.0, 1.012291], 456: [274.0, 1.012291], 330: [269.0, 1.012291], 588: [405.0, 1.0297443], 462: [283.0, 1.012291], 336: [270.0, 1.012291], 594: [340.0, 1.0471976], 468: [289.0, 1.012291], 342: [270.0, 1.012291], 216: [143.0, 1.0297443], 612: [328.0, 1.0471976], 474: [293.0, 1.012291], 348: [267.0, 1.012291], 222: [152.0, 1.0297443], 480: [293.0, 1.012291], 354: [259.0, 1.012291], 228: [161.0, 1.0297443], 486: [285.0, 1.012291], 360: [248.0, 1.012291], 234: [173.0, 1.012291], 492: [273.0, 1.012291], 366: [236.0, 1.012291], 240: [181.0, 1.0297443], 498: [262.0, 1.012291], 372: [227.0, 1.0297443], 246: [194.0, 1.0297443], 504: [254.0, 1.012291], 378: [220.0, 1.0297443], 252: [214.0, 1.012291], 618: [386.0, 1.0297443], 510: [251.0, 1.012291]}")
# Get the endpoints of a line from the cv2.HoughLines function.
#
# 	@param r ------ The given "r" value by cv2.HoughLines.
# 	@param theta -- The given "theta" value by cv2.HoughLines.
#
#	@return ------- An array containing a tuple for each endpoint.
#
def getPolarEndpoints(r, theta):
	return [
		(   # First point coordinates.
			int(np.cos(theta) * r - 1600 * np.sin(theta)),
			int(np.sin(theta) * r + 1600 * np.cos(theta))
		),
		(   # Second point coordinates.
			int(np.cos(theta) * r + 1600 * np.sin(theta)),
			int(np.sin(theta) * r - 1600 * np.cos(theta))
		)
	]

# Convert coordinates from polar to cartesian.
#
# 	@param r ------ The given "r" value by cv2.HoughLines.
# 	@param theta -- The given "theta" value by cv2.HoughLines.
#
#	@return ------- An object containing the slope and y-intercept of the line.
#
def polarToCartesian(r, theta):
	endpoints = getPolarEndpoints(r, theta)
	#print(endpoints)
	m = 0
	if (endpoints[1][0] - endpoints[0][0]) != 0:
		m = (endpoints[1][1] - endpoints[0][1]) / (endpoints[1][0] - endpoints[0][0])

	# Returns the m and b values for the equation y = mx + b.
	return {
		"m": m,
		"b": endpoints[0][1] - endpoints[0][0] * m
	}

def getLineDiff(line1, line2):
	diffSum = 0
	b1 = 306 - line1["b"]
	b2 = 306 - line2["b"]

	line1Diff = (line1["m"] / 2) * (1280 ** 2) + b1 * 1280
	line2Diff = (line2["m"] / 2) * (1280 ** 2) + b2 * 1280
	diffSum += abs(line2Diff - line1Diff)

	return diffSum


arra = []
"""
sums = [0 for i in range(len(goodStr))]
prevLine = None
index = 1
for i in sorted(dictionaryStr):
	if type(prevLine) == type(None):
		prevLine = polarToCartesian(dictionaryStr[i][0], dictionaryStr[i][1])
		next
	if i > goodStr[index]:
		index += 1
	cartesian = polarToCartesian(dictionaryStr[i][0], dictionaryStr[i][1])
	differential = getLineDiff(prevLine, cartesian)
	sums[index] += differential
	arra.append((i, differential))
	prevLine = cartesian
	"""
sums = [0 for i in range(len(goodStr))]
prevLine = None
for i in range(1, len(sums) - 1):
	if type(prevLine) == type(None):
		prevLine = polarToCartesian(dictionaryStr[goodStr[i]][0], dictionaryStr[goodStr[i]][1])
		next
	cartesian = polarToCartesian(dictionaryStr[goodStr[i]][0], dictionaryStr[goodStr[i]][1])
	differential = getLineDiff(prevLine, cartesian)
	sums[i] = differential
	prevLine = cartesian
sumpoints = []
for i in range(1, len(sums)):
	sumpoints.append((goodStr[i], sums[i] / (goodStr[i] - goodStr[i - 1])))
print(sumpoints)
