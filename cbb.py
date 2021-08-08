import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from memory_profiler import profile

def getLocalMaxMin(bins):
	prevVal = 0
	maxima = []
	minima = []

	for i in range(len(bins)):
		nextVal = bins[i]
		if i != len(bins) - 1:
			nextVal = bins[i + 1]

		if prevVal < bins[i]:
			if nextVal < bins[i]:
				maxima.append(i)
		else:
			if prevVal > bins[i] and nextVal > bins[i]:
				minima.append(i)

		prevVal = bins[i]

	return [maxima, minima]

def getClosestVal(arr, val):
	if len(arr) == 0:
		return 0

	minDist = abs(val - arr[0])
	minIndex = 0

	for i in range(1, len(arr)):
		if abs(val - arr[i]) < minDist:
			minDist = abs(val - arr[i])
			minIndex = i

	return minIndex

# Get the endpoints of a line from the cv2.HoughLines function.
#
# 	@param r    The given "r" value by cv2.HoughLines.
# 	@param theta    The given "theta" value by cv2.HoughLines.
# 	@param width    The width of the image.
# 	@param height    The height of the image.
#
#	@return An array containing an object for each endpoint.
#
def getCartesianEndpoints(r, theta, width, height):
	# Get a cartesian representation of our line.
	cartesianValues = polarToCartesian(r, theta)

	# Set up the endpoints of our line.
	firstPoint = {
		"x": 0,
		"y": cartesianValues["b"]
	}
	secondPoint = {}

	# Calculate where the second point lies.
	if cartesianValues["m"] != 0 and cartesianValues["b"] / cartesianValues["m"] <= 0:
		# The second point is on the top line of the screen.
		secondPoint = {
			"x": -1 * cartesianValues["b"] / cartesianValues["m"],
			"y": 0
		}
	elif cartesianValues["m"] != 0 and (height - cartesianValues["b"]) / cartesianValues["m"] >= 0:
		# The second point is on the bottom line of the screen.
		secondPoint = {
			"x": (height - cartesianValues["b"]) / cartesianValues["m"],
			"y": height
		}
	else:
		# The second point is on the right end of the screen.
		secondPoint = {
			"x": width,
			"y": cartesianValues["m"] * width + cartesianValues["b"]
		}

	return [firstPoint, secondPoint]


def functionOfX(r, theta, x):
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*r
	y0 = b*r
	x1 = int(x0 + 1600*(-b))
	y1 = int(y0 + 1600*(a))
	x2 = int(x0 - 1600*(-b))
	y2 = int(y0 - 1600*(a))
	m = 100000
	if x2 - x1 != 0:
		m = (y2 - y1) / (x2 - x1)

	return int(y1 + m * (x - x1))


def getCourtColorsFromImage(inputImage):
	width = inputImage.shape[1]
	height = inputImage.shape[0]

	imageYCC = cv2.cvtColor(
		inputImage,
		cv2.COLOR_BGR2YCR_CB
	)

	convolutionYCC = np.zeros([height, width, 1], dtype = np.uint8)

	for x in range(width):
		for y in range(height):
			convolutionYCC.itemset(
				(y, x, 0),
				int((imageYCC.item(y, x, 1) + imageYCC.item(y, x, 2)) / 2)
			)
	
	dataYCC = np.float32(
		convolutionYCC.reshape((-1, 1))
	)

	colorBins = plt.hist(dataYCC, 256, [0, 256])[0]

	maxMinColors = getLocalMaxMin(colorBins)
	centerColor = maxMinColors[0][
		getClosestVal(maxMinColors[0], 124)
	]
	colorRadius = abs(
		centerColor - maxMinColors[1][
			getClosestVal(maxMinColors[1], centerColor)
		]
	)

	plt.close('all')

	return (centerColor, colorRadius, convolutionYCC)

def getCourtEdgesFromImage(inputImage):
	width = inputImage.shape[1]
	height = inputImage.shape[0]

	centerColor, colorRadius, convolutionYCC = getCourtColorsFromImage(inputImage)
	courtEdges = np.zeros([height, width, 1], dtype = np.uint8)
	pixelColumn = 0
	pixelRow = 0

	if colorRadius == 1:
		while pixelColumn < width:
			convolutionItem = convolutionYCC.item(pixelRow, pixelColumn, 0)

			if (abs(convolutionItem - centerColor) < colorRadius
					or convolutionItem == centerColor - colorRadius - 1):
				matches = 0

				maxRow = 10
				if maxRow + pixelRow > height:
					maxRow = height - pixelRow
				for i in range(1, maxRow):
					convolutionItemBelow = convolutionYCC.item(pixelRow + i, pixelColumn, 0)

					if (abs(convolutionItemBelow - centerColor) < colorRadius
							or convolutionItemBelow == centerColor - colorRadius - 1):
						matches += 1

				if matches >= 3:
					courtEdges.itemset((pixelRow, pixelColumn, 0), 220)

				pixelColumn += 1
				pixelRow = 0
			else:
				pixelRow += 1
				if pixelRow >= height:
					pixelRow = 0
					pixelColumn += 1
	else:
		while pixelColumn < width:
			if abs(convolutionYCC.item(pixelRow, pixelColumn, 0) - centerColor) < colorRadius:
				matches = 0

				maxRow = 10
				if maxRow + pixelRow > height:
					maxRow = height - pixelRow
				for i in range(1, maxRow):
					if abs(convolutionYCC.item(pixelRow + i, pixelColumn, 0) - centerColor) < colorRadius:
						matches += 1

				if matches >= 3:
					courtEdges.itemset((pixelRow, pixelColumn, 0), 220)

				pixelColumn += 1
				pixelRow = 0
			else:
				pixelRow += 1
				if pixelRow >= height:
					pixelRow = 0
					pixelColumn += 1

	return courtEdges

def getCourtAreaFromImage(inputImage):
	width = inputImage.shape[1]
	height = inputImage.shape[0]

	centerColor, colorRadius, convolutionYCC = getCourtColorsFromImage(inputImage)
	courtEdges = np.zeros([height, width, 1], dtype = np.uint8)
	pixelColumn = 0
	pixelRow = 0

	if colorRadius == 1:
		while pixelColumn < width:
			convolutionItem = convolutionYCC.item(pixelRow, pixelColumn, 0)
			if (abs(convolutionItem - centerColor) < colorRadius
					or convolutionItem == centerColor - colorRadius - 1):
				courtEdges.itemset((pixelRow, pixelColumn, 0), 150)
			pixelRow += 1
			if pixelRow >= height:
				pixelRow = 0
				pixelColumn += 1
	else:
		while pixelColumn < width:
			if abs(convolutionYCC.item(pixelRow, pixelColumn, 0) - centerColor) < colorRadius:
				courtEdges.itemset((pixelRow, pixelColumn, 0), 150)
			pixelRow += 1
			if pixelRow >= height:
				pixelRow = 0
				pixelColumn += 1

	return courtEdges

# Convert from transformed radians given by cv2.HoughLines to adjusted 
# degrees.
#
# 	@param theta -- The given "theta" value by cv2.HoughLines.
#
#	@return ------- The converted degree value.
#
def radiansToDegrees(theta):
	# Find the equivalent positive angle.
	while theta < 0:
		theta = math.pi * 2 + theta

	# Find the equivalent angle under 2pi radians.
	while theta > math.pi * 2:
		theta = theta - math.pi * 2

	# Translate the angle then return the equivalent degree measure.
	if theta >= 0 and theta <= math.pi / 2:
		return math.degrees(math.pi / 2 - theta)
	else:
		return math.degrees(5 * math.pi / 2 - theta)


# Convert coordinates from polar to cartesian.
#
# 	@param r ------ The given "r" value by cv2.HoughLines.
# 	@param theta -- The given "theta" value by cv2.HoughLines.
#
#	@return ------- An object containing the slope and y-intercept of the line.
#
def polarToCartesian(r, theta):
	endpoints = getPolarEndpoints(r, theta)
	m = 0
	if (endpoints[1][0] - endpoints[0][0]) != 0:
		m = (endpoints[1][1] - endpoints[0][1]) / (endpoints[1][0] - endpoints[0][0])

	# Returns the m and b values for the equation y = mx + b.
	return {
		"m": m,
		"b": endpoints[0][1] - endpoints[0][0] * m
	}


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


# Draw a line on an image from polar coordinates.
#
#	@param image ------ The image to draw onto.
# 	@param r ---------- The given "r" value by cv2.HoughLines.
# 	@param theta ------ The given "theta" value by cv2.HoughLines.
# 	@param color ------ The color to draw the line with.
# 	@param thickness -- The desired thickness of the line.
#
def drawLine(image, r, theta, color, thickness):
	endpoints = getPolarEndpoints(r, theta)

	# Draw the line on the image.
	cv2.line(
		image,
		(endpoints[0][0], endpoints[0][1]),
		(endpoints[1][0], endpoints[1][1]),
		color,
		thickness
	)

def getTopLine(inputImage):
	width = inputImage.shape[1]
	height = inputImage.shape[0]

	courtEdges = getCourtEdgesFromImage(inputImage)

	sChannelFromHSVImage = cv2.split(cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV))[1]
	
	lines = cv2.HoughLines(courtEdges, 5, np.pi/180, 30)
	topLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 350, 430, apertureSize = 3)
	topLineMask = np.zeros([height, width, 1], dtype = np.uint8)

	mainR = 0
	mainTheta = 0
	linesIncluded = 0

	if type(lines) != type(None):
		drawLine(topLineMask, lines[0][0][0], lines[0][0][1], 255, 30)

	for x in range(width):
		for y in range(height):
			if topLineMask.item(y, x, 0) != 255:
				topLineCannyDetection.itemset((y, x), 0)
	
	lines = cv2.HoughLines(topLineCannyDetection, 5, np.pi/180, 30)

	if type(lines) != type(None):
		for i in range(2):
			mainR += lines[i][0][0]
			mainTheta += lines[i][0][1]
			linesIncluded += 1

	if linesIncluded != 0:
		return (mainR / linesIncluded, mainTheta / linesIncluded)
	else:
		return (0, 0)

def getLineIntersection(r1, theta1, r2, theta2):
	cartesianVals1 = polarToCartesian(r1, theta1)
	cartesianVals2 = polarToCartesian(r2, theta2)
	m1 = cartesianVals1["m"]
	b1 = cartesianVals1["b"]
	m2 = cartesianVals2["m"]
	b2 = cartesianVals2["b"]

	x = -100
	if m1 - m2 != 0:
		x = int((b2 - b1) / (m1 - m2))
	y = int(m1 * x + b1)

	return (x, y)


def displayImage(title, image):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#@profile
def detectCourtEdges(image, displayResult = True, shouldCrop = False, analyze = False):
	inputImage = image
	if shouldCrop:
		inputImage = image[(int(image.shape[0] * 0.375)):(int(image.shape[0] * 0.8)), 0:image.shape[1]]

	width = inputImage.shape[1]
	height = inputImage.shape[0]

	outputImage = inputImage.copy()
	#start = time.time()

	topLineR, topLineTheta = getTopLine(inputImage)

	sChannelFromHSVImage = cv2.split(cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV))[1]

	if topLineR != 0 and topLineTheta != 0:
		#print("Top line dimensions: R: {}, Theta: {}".format(topLineR, topLineTheta))
		drawLine(outputImage, topLineR, topLineTheta, (0, 255, 0), 2)

		cartesianValues = polarToCartesian(topLineR, topLineTheta)

		for x in range(width):
			cutOff = cartesianValues["m"] * x + cartesianValues["b"] + 3

			for y in range(height):
				if y < cutOff:
					sChannelFromHSVImage.itemset((y, x), 0)

	endLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 350, 430, apertureSize = 3)

	lines = cv2.HoughLines(endLineCannyDetection, 1, np.pi/180, 50)

	endLineR = 0
	endLineTheta = 0

	if type(lines) != type(None):
		for line in lines:
			r = line[0][0]
			theta = line[0][1]

			if abs(topLineTheta - theta) > math.radians(25):
				#print("End line dimensions: R: {}, Theta: {}".format(r, theta))
				#drawLine(outputImage, r, theta, (0, 255, 0), 2)
				endLineR = r
				endLineTheta = theta

				cartesianValues = polarToCartesian(r, theta)

				for x in range(width):
					cutOff = cartesianValues["m"] * x + cartesianValues["b"] + 3

					for y in range(height):
						if y < cutOff:
							sChannelFromHSVImage.itemset((y, x), 0)

				break

	freeThrowLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 250, 330, apertureSize = 3)

	lines = cv2.HoughLines(freeThrowLineCannyDetection, 1, np.pi/180, 50)

	freeThrowLineVals = []

	if type(lines) != type(None):
		i = 0
		for line in lines:
			r = line[0][0]
			theta = line[0][1]

			if (abs(topLineTheta - theta) < math.radians(2)
					and abs(topLineR - r) > 20):
				#print("Free throw line dimensions: R: {}, Theta: {}".format(r, theta))
				#drawLine(outputImage, r, theta, (0, 255, 0), 2)
				freeThrowLineVals.append([r, theta])

				if i > 0:
					break
				else:
					i += 1

	#print(time.time() - start)
	if analyze:
		tupleVal = (outputImage, [topLineR, topLineTheta], [endLineR, endLineTheta])
		for i in freeThrowLineVals:
			tupleVal = tupleVal + (i,)
		for i in range(5 - len(tupleVal)):
			tupleVal = tupleVal + ([0, 0],)
		return tupleVal

	if displayResult:
		displayImage("College Basketball", outputImage)
	else:
		return outputImage

def analyzeDetection(inputImage, crop = False):
	outputImage, topLine, endLine, fr1, fr2 = detectCourtEdges(inputImage, analyze = True, displayResult = False, shouldCrop = crop)
	x1, y1 = getLineIntersection(topLine[0], topLine[1], endLine[0], endLine[1])
	x2, y2 = getLineIntersection(fr1[0], fr1[1], endLine[0], endLine[1])
	x3, y3 = getLineIntersection(fr2[0], fr2[1], endLine[0], endLine[1])

	return (outputImage, [topLine, endLine, fr1, fr2, [x1, y1], [x2, y2], [x3, y3]])

def analyzeVideo(inputVideo, createOutput = False, skip = 0, initialSkip = 180):
	frameInfo = {}
	frames = {}
	analyzedFrames = {}

	firstRead = True
	frameIndex = 0
	while inputVideo.isOpened():
		frame = inputVideo.read()[1]
		frames[frameIndex] = frame

		if firstRead:
			for i in range(initialSkip):
				frame = inputVideo.read()[1]
			firstRead = False

		for i in range(skip):
			frame = inputVideo.read()

		imageResult, detection = analyzeDetection(frame, crop = True)
		frameInfo[frameIndex] = detection
		analyzedFrames[frameIndex] = imageResult

		cv2.imshow('frame', imageResult)
		frameIndex += 1
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	offset = int(frames[0].shape[0] * 0.375)
	frame2 = inputVideo.read()[1]
	#frame2 = frame2[(int(frame2.shape[0] * 0.375)):(int(frame2.shape[0] * 0.8)), 0:frame2.shape[1]]
	video = cv2.VideoWriter('enhancedDetection2.mp4', -1, 1, (frame2.shape[1], frame2.shape[0]))
	for frame in frameInfo:
		cv2.circle(frames[frame], (frameInfo[frame][4][0], frameInfo[frame][4][1] + offset), 5, (0,0,255), -1)
		cv2.circle(frames[frame], (frameInfo[frame][5][0], frameInfo[frame][5][1] + offset), 5, (0,0,255), -1)
		cv2.circle(frames[frame], (frameInfo[frame][6][0], frameInfo[frame][6][1] + offset), 5, (0,0,255), -1)
		video.write(frames[frame])
		#cv2.imshow('frame', frames[frame])
		#time.sleep(1)
		
		#if cv2.waitKey(1) & 0xFF == ord('q'):
		#	break
	video.release()

	"""
	if createOutput:
		frame2 = inputVideo.read()[1]
		frame2 = frame2[(int(frame2.shape[0] * 0.375)):(int(frame2.shape[0] * 0.8)), 0:frame2.shape[1]]
		video = cv2.VideoWriter('enhancedDetection.mp4', -1, 1, (frame2.shape[1], frame2.shape[0]))
		video.release()
	"""

#imgReg = cv2.imread("gameShot.png", 1)

#detectCourtEdges(imgReg)
#cap = cv2.VideoCapture('illiniClip.mp4')
#analyzeVideo(cap)

def detectLineElements(detection):
	height, width = detection.shape[0:2]
	lines = np.zeros(detection.shape, dtype = np.uint8)
	lineElements = []
	pixels = []
	for x in range(width):
		for y in range(height):
			if (detection.item(y, x) == 255 and y > 0 and y < height - 1
					and detection.item(y - 1, x) == 0 and detection.item(y + 1, x) == 0):
				pixels.append((y, x))
	deadPixels = []
	for i in pixels:
		if (i[0], i[1]) not in deadPixels:
			length = 1
			dp = [(i[0], i[1])]
			while i[1] + length < width and detection.item(i[0], i[1] + length) == 255:
				dp.append((i[0], i[1] + length))
				length += 1
			l2 = 1
			while i[1] - l2 >= 0 and detection.item(i[0], i[1] - l2) == 255:
				dp = [(i[0], i[1] - l2)] + dp
				l2 += 1
			if length + l2 - 1 > 3:
				lineElements.append([dp, length + l2 - 1])
			deadPixels += dp

	for i in lineElements:
		for j in i[0]:
			lines.itemset(j, 255)

	return lines

def gradientify(mask, lineImage, spread):
	height, width = mask.shape[0:2]

	lHeight = 0
	xc = 0
	while lHeight == 0 and xc < width:
		for y in range(height):
			if mask.item(y, xc) != 0:
				lHeight += 1
		xc += 1

	#grad = np.zeros([lHeight, width], dtype = np.uint8)
	#gradDict = {}
	#mlw = 0
	#minlw = 0

	weight = 0

	for x in range(width):
		for y in range(height):
			if mask.item(y, x) != 0:
				pixelsToCheck = []
				while len(pixelsToCheck) < lHeight and mask.item(y + len(pixelsToCheck), x) != 0:
					pixelsToCheck.append((y + len(pixelsToCheck), x))

				#lineWeight = 0
				for i in range(len(pixelsToCheck)):
					topI = i - spread
					bottomI = i + spread

					if topI < 0:
						topI = 0
					if bottomI >= len(pixelsToCheck):
						bottomI = len(pixelsToCheck) - 1

					totalVal = 0
					nPixels = 0
					for j in range(topI, bottomI + 1):
						totalVal += lineImage.item(pixelsToCheck[j])
						nPixels += 1

					if nPixels != 0 and totalVal / nPixels >= 255 / (2 * spread + 1) - 1:
						#grad.itemset((i, x), totalVal / nPixels)
						dMultVal = abs(i - lHeight / 2)
						if dMultVal <= spread + 1:
							dMultVal = 8
						elif dMultVal <= 2 * spread + 1:
							dMultVal = 1
						elif dMultVal <= 3 * spread + 1:
							dMultVal = -6
						elif dMultVal < 4 * spread + 1:
							dMultVal = -16
						else:
							dMultVal = -4
						weight += dMultVal
						#lineWeight += dMultVal
				#if mlw < lineWeight:
				#	mlw = lineWeight
				#if minlw > lineWeight:
				#	minlw = lineWeight
				#gradDict[x] = lineWeight
				break
	"""
	iPoints = {}

	leftAverage = 0
	nPoints = 1
	delay = 0
	for x in range(width):
		cVal = gradDict[x]
		greater = cVal >= (leftAverage / nPoints)
		if delay > 0 and greater:
			if delay > 5:
				iPoints[x - delay] = [delay, leftAverage / nPoints]
			delay = 0
		if not greater:
			delay += 1

		leftAverage += cVal
		nPoints += 1

	print(iPoints)
	print(len(iPoints))
	print(weight)
	"""
	#grad2 = np.zeros([lHeight, width], dtype = np.uint8)
	#topval = mlw - minlw
	#for x in range(width):
	#	val = gradDict[x] - minlw
	#	colVal = int((val / topval) * 255)
	#	for y in range(lHeight):
	#		grad2.itemset((y, x), colVal)

	#print(gradDict)
	#displayImage(str(weight), grad2)
	#displayImage("gr", lineImage)
	return weight

def getLineDiff(line1, line2, width, height):
	b1 = height - line1["b"]
	b2 = height - line2["b"]

	line1Diff = (line1["m"] / 2) * (width ** 2) + b1 * width
	line2Diff = (line2["m"] / 2) * (width ** 2) + b2 * width

	return abs(line2Diff - line1Diff)

def analyzeEndLineDetections(elGroups, endLineDict, cannyDict, offset, fullFrameDict, fpr):
	newElGroups = []
	firstFrame = min(fullFrameDict.keys())
	lastFrame = max(fullFrameDict.keys())
	height, width = cannyDict[firstFrame].shape[0:2]

	for i in range(len(elGroups)):
		tempGroup = {}
		keys = sorted(elGroups[i].keys())
		start = keys[0] - fpr * 4
		end = keys[-1] + fpr * 4

		if start < firstFrame:
			start = firstFrame
		if end > lastFrame:
			end = lastFrame

		for j in range(start, end + 1, fpr):
			tempGroup[j] = endLineDict[j]
		newElGroups.append(tempGroup)

	print(sorted(fullFrameDict.keys()))

	yLinesModel = [
		43.48837209302326,
		43.48837209302326,
		28.297555158020273,
		20.527728085867622,
		31.323792486583184,
		47.59391771019678,
		94.88372093023256,
		146.88729874776385,
		119.5169946332737,
		76.19558735837806,
		90.79308288610615,
		137.91592128801432,
		133.0649970184854,
		73.74776386404294,
		19.007155635062613,
		4.257602862254025,
		3.953488372093023
	]

	#for jj in range(17):
		#yLines[jj] = []

	finalElGroups = []

	for group in newElGroups:
		tempGroup = {}

		for i in group.keys():
			gradeFrame = cannyDict[i]
			endVals = getPolarEndpoints(group[i][0], group[i][1])
			endLineMask = np.zeros([height, width], dtype = np.uint8)
			cv2.line(endLineMask, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 30)
			gradeFrame[endLineMask < 255] = 25

			cartVals = polarToCartesian(fullFrameDict[i][0], fullFrameDict[i][1])
			m = cartVals["m"]
			b = cartVals["b"]

			for x in range(width):
				for y in range(height):
					if gradeFrame.item(y, x) != 25 and y < m*x + b + offset:
						gradeFrame.itemset((y, x), 25)

			lVals = []
			lWidth = 0
			startY = 0
			endY = 0
			yc = 0
			while yc < height:
				stop = False
				for x in range(width):
					if gradeFrame.item(yc, x) != 25:
						if x == 0:
							stop = True
							break
						lWidth += 1
				if stop:
					endY = yc
					break
				if lWidth != 0:
					if len(lVals) == 0:
						startY = yc
					lVals.append(lWidth)
					lWidth = 0
				yc += 1

			if endY <= startY:
				continue

			lWidth = max(lVals)
			#print(lVals)
			#print("lWidth:" + str(lWidth))
			#print("yc:" + str(yc))

			#cv2.imshow('frame', gradeFrame)
			#if cv2.waitKey(0) & 0xFF == ord('q'):
			#	break
			#else:
			#	cv2.destroyAllWindows()
			"""
			cv2.imshow('frame', gradeFrame)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
			else:
				cv2.destroyAllWindows()"""

			dZone = np.zeros([lWidth, endY - startY], dtype = np.uint8)

			for y in range(height):
				if endY - y < 0:
					break
				for x in range(width):
					if gradeFrame.item(y, x) != 25:
						#print("y: " + str(y))
						#print("height: " + str(height))
						#print("length: " + str(height + 1 - finalY))
						pixelsToCheck = []
						while len(pixelsToCheck) < lWidth and gradeFrame.item(y, x + len(pixelsToCheck)) != 25:
							pixelsToCheck.append((y, x + len(pixelsToCheck)))
							dZone.itemset((len(pixelsToCheck) - 1, endY - y - 1), gradeFrame.item(y, x + len(pixelsToCheck)))
						break

			

			lines = cv2.HoughLines(dZone, 1, np.pi/360, int((endY - startY) / 3))

			endLineR = 0
			endLineTheta = 0
			num = 0

			#dZone = np.zeros([lWidth, endY - startY], dtype = np.uint8)
			lineDetectDict = {}

			if type(lines) != type(None):
				kt = 0
				for line in lines:
					if kt > 7:
						break
					r = line[0][0]
					theta = line[0][1]
					endLineR += r
					endLineTheta += theta
					num += 1
					kt += 1

			endLineR /= num
			endLineTheta /= num

			if type(lines) != type(None):
				kt = 0
				for line in lines:
					if kt > 7:
						break
					r = line[0][0]
					theta = line[0][1]
					lineDetectDict[abs(1 - endLineR / r) + abs(1 - endLineTheta / theta)] = [r, theta]
					kt += 1

			endLineR = 0
			endLineTheta = 0
			num = 0

			for nk in sorted(lineDetectDict.keys()):
				if num > 3:
					break
				endLineR += lineDetectDict[nk][0]
				endLineTheta += lineDetectDict[nk][1]
				num += 1
			
			endLineR /= num
			endLineTheta /= num

			dZone2 = np.zeros([17, endY - startY], dtype = np.uint8)
			dZone3 = np.zeros([17, endY - startY], dtype = np.uint8)

			cartVals = polarToCartesian(endLineR, endLineTheta)
			m = cartVals["m"]
			b = cartVals["b"]

			yLines = {}
			for jj in range(17):
				yLines[jj] = []

			for x in range(endY - startY):
				for y in range(17):
					newYTemp = m*x + b + y - 8
					if newYTemp >= 0 and newYTemp < lWidth:
						dZone2.itemset((y, x), dZone.item(int(newYTemp), x))
						yLines[y].append(dZone.item(int(newYTemp), x))

			xMeanSums = []

			for x in range(endY - startY):
				meanPos = 0
				totalVals = 0
				for y in range(17):
					if (dZone2.item(y, x) == 255):
						meanPos += y
						totalVals += 1
				if totalVals != 0:
					dZone3.itemset((int(meanPos / totalVals), x), 255)
					xMeanSums.append(meanPos / totalVals)

			meanLineY = sum(xMeanSums) / len(xMeanSums)

			diffSum = 0

			for mVal in xMeanSums:
				diffSum += abs(int(mVal - meanLineY)) - 3

			for mVal2 in range(0, (endY - startY) - len(xMeanSums)):
				diffSum += 8


			"""yCounts = {}
			for jj in yLines.keys():
				yCounts[jj] = 0
				for kk in yLines[jj]:
					yCounts[jj] += kk
				if len(yLines[jj]) > 0:
					yCounts[jj] /= len(yLines[jj])
					diffSum += abs(yLinesModel[jj] - yCounts[jj])
				else:
					diffSum += 10000"""

			print("Score : " + str(diffSum) + " mlY: " + str(meanLineY))

			if diffSum < 100:
				tempGroup[i] = group[i]

			#drawLine(dZone2, 9, np.pi/2, 255, 2)

			"""cv2.imshow('frame', dZone2)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
			else:
				cv2.destroyAllWindows()

			cv2.imshow('frame2', dZone3)
			if cv2.waitKey(0) & 0xFF == ord('q'):
				break
			else:
				cv2.destroyAllWindows()"""
		finalElGroups.append(tempGroup)



	#for ln in yLines.keys():
	#	sumVal = 0
	#	line = yLines[ln]
	#	for val in line:
	#		sumVal += val
	#	print("Line " + str(ln) + " avg : " + str(sumVal / len(line)))

	return finalElGroups


					








def experimentVideo(inputVideoPath):
	inputVideo = cv2.VideoCapture(inputVideoPath)
	framesPerRead = 5
	firstRead = True
	frameIndex = 1
	topLineDict = {}
	endLineDict = {}
	cannyDict = {}
	height, width = (0, 0)
	offset = 0
	#video = None
	while inputVideo.isOpened() and frameIndex < 600:
		frame = inputVideo.read()[1]

		if firstRead:
			for i in range(400):
				frame = inputVideo.read()[1]
				frameIndex += 1
			frame2 = frame[(int(frame.shape[0] * 0.375)):(int(frame.shape[0] * 0.8)), 0:frame.shape[1]]
			#video = cv2.VideoWriter('goodLineDetection2.avi', -1, 1, (frame2.shape[1], frame2.shape[0]))
			#print(frame.shape)
			firstRead = False

		for i in range(framesPerRead - 1):
			frame = inputVideo.read()[1]
			frameIndex += 1

		if offset == 0:
			offset = int(frame.shape[0] * 0.375)

		imageResult, detection = analyzeDetection(frame, crop = True)
		tl = detection[0]
		height, width = frame.shape[0:2]
		#print(frame.shape)
		endVals = getPolarEndpoints(tl[0], tl[1])
		topLineMask = np.zeros([height, width], dtype = np.uint8)
		cv2.line(topLineMask, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 30)
		sChannelFromHSVImage = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[1]
		topLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 350, 430, apertureSize = 3)
		tl3 = topLineCannyDetection.copy()
		topLineCannyDetection[:] = topLineCannyDetection[:] / 8
		#topLineCannyDetection[topLineCannyDetection == 255] = 100
		tl2 = topLineCannyDetection.copy()
		
		tl2[topLineMask < 255] = 0

		for x in range(width):
			for y in range(height):
				if tl2.item(y, x) != 0 and y > 0 and y < height - 1:
					if tl2.item(y - 1, x) == 0 and tl2.item(y + 1, x) == 0:
						tl2.itemset((y, x), 255)

		nf = detectLineElements(tl2)
		topLineCannyDetection[nf == 255] = 255

		gra = gradientify(topLineMask, topLineCannyDetection, 2)
		print(gra, frameIndex)

		#endLineMask = np.zeros([height, width], dtype = np.uint8)
		#cv2.line(endLineMask, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 30)

		#displayImage(str(gra), topLineCannyDetection)
		endLineDict[frameIndex] = detection[1]
		cannyDict[frameIndex] = tl3
		#endLine = detection[1]
		#endVals = getPolarEndpoints(endLine[0], endLine[1])
		
		#sChannelFromHSVImage = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[1]
		##topLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 350, 430, apertureSize = 3)
		#topLineCannyDetection[:] = topLineCannyDetection[:] / 8
		#topLineCannyDetection[topLineCannyDetection == 255] = 100
		
		#tl3[endLineMask < 255] = 0

		#for x in range(width):
		#	for y in range(height):
		#		if tl3.item(y, x) != 0 and y > 0 and y < height - 1:
		#			if tl3.item(y - 1, x) == 0 and tl3.item(y + 1, x) == 0:
		#				tl3.itemset((y, x), 255)

		#nf = detectLineElements(tl2)
		#topLineCannyDetection[nf == 255] = 255

		#gra = gradientify(topLineMask, topLineCannyDetection, 2)

		#cv2.imshow('frame', tl3)
		#if cv2.waitKey(0) & 0xFF == ord('q'):
		#	break
		#else:
		#	cv2.destroyAllWindows()


		if gra > 4000:
			topLineDict[frameIndex] = detection[0]

			#cv2.destroyAllWindows()
			#cv2.imshow(str(gra), imageResult)

		#time.sleep(1)
		frameIndex += 1
		
		#if cv2.waitKey(1) & 0xFF == ord('q'):
			#break
	
	weightedFrames = {}
	#print(sorted(topLineDict))
	prevLine = None
	prevI = None
	for i in sorted(topLineDict):
		#print(i, prevI)
		if type(prevLine) == type(None):
			prevLine = polarToCartesian(topLineDict[i][0], topLineDict[i][1])
			prevI = i
			weightedFrames[i] = topLineDict[i]
			continue
		cartesian = polarToCartesian(topLineDict[i][0], topLineDict[i][1])
		differential = getLineDiff(prevLine, cartesian, width, int(height * 0.425))
		
		differentialPerFrame = differential / (i - prevI)

		detectionWeight = (1 / 3) ** (differentialPerFrame / 2000)

		weightedR = weightedFrames[prevI][0] + (topLineDict[i][0] - weightedFrames[prevI][0]) * detectionWeight
		weightedTheta = weightedFrames[prevI][1] + (topLineDict[i][1] - weightedFrames[prevI][1]) * detectionWeight

		weightedFrames[i] = [weightedR, weightedTheta]

		prevLine = cartesian
		prevI = i

	topLineDict = weightedFrames

	print(endLineDict)

	endLineFrames = {}
	elGroups = []
	#print(sorted(topLineDict))
	prevLine = None
	prevI = None
	goodDetections = 0
	badDetections = 0
	isEndCourt = False
	for i in sorted(endLineDict):
		#print(i, prevI)
		if type(prevLine) == type(None):
			prevLine = polarToCartesian(endLineDict[i][0], endLineDict[i][1])
			prevI = i
			continue
		cartesian = polarToCartesian(endLineDict[i][0], endLineDict[i][1])
		differential = getLineDiff(prevLine, cartesian, width, int(height * 0.425))
		
		differentialPerFrame = differential / (i - prevI)

		if isEndCourt:
			if differentialPerFrame < 10000:
				badDetections = 0
			else:
				badDetections += 1

			if badDetections > 4:
				goodDetections = 0
				badDetections = 0
				#stops.append(i - 15)

				for j in range(1, 5):
					try:
						del endLineFrames[i - framesPerRead * j]
					except:
						pass
				elGroups.append(endLineFrames)
				endLineFrames = {}
				isEndCourt = False
		else:
			if differentialPerFrame < 10000:
				goodDetections += 1
			else:
				goodDetections = 0

			if goodDetections > 4:
				isEndCourt = True
				goodDetections = 0
				badDetections = 0

		if isEndCourt:
			endLineFrames[i] = endLineDict[i]

		
		#detectionWeight = (1 / 2) ** (differentialPerFrame / 2500)



		#weightedR = weightedFrames[prevI][0] + (topLineDict[i][0] - weightedFrames[prevI][0]) * detectionWeight
		#weightedTheta = weightedFrames[prevI][1] + (topLineDict[i][1] - weightedFrames[prevI][1]) * detectionWeight

		#weightedFrames[i] = [weightedR, weightedTheta]
		

		prevLine = cartesian
		prevI = i

	if len(endLineFrames) != 0:
		elGroups.append(endLineFrames)
		endLineFrames = {}



	inputVideo = cv2.VideoCapture(inputVideoPath)
	print("End Line Periods: ", elGroups)
	#if len(elGroups) != 0:
	#	endLineFrames = elGroups[0]

	fullFrameDict = {}
	fullFrameDict2 = {}
	
	firstFrame = True
	prevFrame = True
	for i in sorted(topLineDict):
		if firstFrame == True:
			firstFrame = i
			prevFrame = i
		else:
			diff = i - prevFrame
			r1, theta1 = topLineDict[prevFrame]
			r2, theta2 = topLineDict[i]
			rDiff = r2 - r1
			thetaDiff = theta2 - theta1
			for j in range(prevFrame, diff + prevFrame):
				fullFrameDict[j] = [r1 + rDiff * ((j - prevFrame) / diff), theta1 + thetaDiff * ((j - prevFrame) / diff)]

			prevFrame = i

	maxIndexFFDict = max(topLineDict.keys())
	fullFrameDict[maxIndexFFDict] = topLineDict[maxIndexFFDict]

	elGroups = analyzeEndLineDetections(elGroups, endLineDict, cannyDict, offset, fullFrameDict, framesPerRead)

	
	firstFrame = True
	prevFrame2 = True
	for frameGroup in elGroups:
		endLineFrames = frameGroup
		firstFrame = True
		prevFrame2 = True
		for i in sorted(endLineFrames):
			if firstFrame == True:
				firstFrame = i
				prevFrame2 = i
			else:
				diff = i - prevFrame2
				r1, theta1 = endLineFrames[prevFrame2]
				r2, theta2 = endLineFrames[i]
				rDiff = r2 - r1
				thetaDiff = theta2 - theta1
				for j in range(prevFrame2, diff + prevFrame2):
					fullFrameDict2[j] = [r1 + rDiff * ((j - prevFrame2) / diff), theta1 + thetaDiff * ((j - prevFrame2) / diff)]
				prevFrame2 = i

	#print(fullFrameDict, prevFrame)

	frame = inputVideo.read()[1]
	video = cv2.VideoWriter('smoothed71.mp4', -1, 30.0, (frame.shape[1], frame.shape[0]))
	for i in range(firstFrame):
		frame = inputVideo.read()[1]
	offset = int(frame.shape[0] * 0.375)
	height, width = frame.shape[0:2]

	for i in range(firstFrame, prevFrame):
		endVals = getPolarEndpoints(fullFrameDict[i][0], fullFrameDict[i][1])
		cv2.line(frame, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), (0, 255, 0), 2)
		try:
			endVals2 = getPolarEndpoints(fullFrameDict2[i][0], fullFrameDict2[i][1])
			cv2.line(frame, (endVals2[0][0], endVals2[0][1] + offset), (endVals2[1][0], endVals2[1][1] + offset), (255, 0, 0), 2)
		except:
			pass

		## START INSERT
		#offset = int(frame.shape[0] * 0.375)

		imageResult, detection = analyzeDetection(frame, crop = True)
		
		#print(frame.shape)
		#endVals = getPolarEndpoints(tl[0], tl[1])
		topLineMask8 = np.zeros([height, width], dtype = np.uint8)
		cv2.line(topLineMask8, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 30)
		sChannelFromHSVImage8 = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[1]
		topLineCannyDetection8 = cv2.Canny(sChannelFromHSVImage8, 38000, 50000, apertureSize = 7)
		topLineCannyDetection8[:] = topLineCannyDetection8[:] / 2
		#topLineCannyDetection[topLineCannyDetection == 255] = 100
		#tl2 = topLineCannyDetection.copy()
		topLineCannyDetection8[topLineMask8 < 255] = 0
		intersections = np.zeros([height, width], dtype = np.uint8)
		cv2.line(intersections, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 1)
		extraMask = np.zeros([height, width], dtype = np.uint8)

		
		lines = cv2.HoughLines(topLineCannyDetection8,1,np.pi/180,20)
		newLines = []
		for line in reversed(lines):
			for rho,theta in line:
				if not (abs(fullFrameDict[i][1] - theta) > math.radians(25)):
					continue
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))

				#cv2.line(topLineCannyDetection,(x1,y1),(x2,y2),rank,1)
				newLines.append([(x1,y1),(x2,y2)])
		rank = 255 - len(newLines) * 51
		for line in newLines:
			rank += 64
			if rank > 0:
				cv2.line(extraMask,line[0],line[1],rank,1)
		extraMask[intersections == 0] = 0

		circling = False
		circleInProgressX = 0
		circleInProgressY = 0
		circleInProgressWeight = 0
		circleVals = 0
		circles = []
		dist = 0
		for x in range(width):
			if circling:
				dist += 1
			for y in range(height):
				if extraMask.item(y,x) != 0:
					if circling:
						if dist >= 20:
							circles.append([(int(circleInProgressX / (circleInProgressWeight)), 
							int(circleInProgressY / (circleInProgressWeight))), circleInProgressWeight / circleVals])
							circling = False
							circleInProgressX = 0
							circleInProgressY = 0
							circleInProgressWeight = 0
							circleVals = 0
							dist = 0
							break
						cValue = extraMask.item(y, x)
						circleInProgressX += x * cValue
						circleInProgressY += y * cValue
						circleInProgressWeight += cValue
						circleVals += 1
					else:
						circling = True
						cValue = extraMask.item(y, x)
						circleInProgressX += x * cValue
						circleInProgressY += y * cValue
						circleInProgressWeight += cValue
						circleVals += 1
					break

					#cv2.circle(frame,(x,y),int(5 * (extraMask.item(y,x) / 255)),(255, 50, 255),-1)
		if circleVals != 0:
			circles.append([(int(circleInProgressX / (circleInProgressWeight)), 
							int(circleInProgressY / (circleInProgressWeight))), circleInProgressWeight / circleVals])

		for circle in circles:
			cv2.circle(frame, circle[0], 2, (140, 0, 255), -1)

		## END INSERT

		video.write(frame)
		frame = inputVideo.read()[1]

	video.release()

cap = cv2.VideoCapture('illiniClip.mp4')
experimentVideo("illiniClip.mp4")

def experiment(image):
	width = image.shape[1]
	height = image.shape[0]

	imageYCC = cv2.cvtColor(
		image,
		cv2.COLOR_BGR2YCR_CB
	)

	imageHSV = cv2.cvtColor(
		image,
		cv2.COLOR_BGR2LAB
	)

	l, a, b = cv2.split(imageHSV)

	h, s, v = cv2.split(cv2.cvtColor(
		image,
		cv2.COLOR_BGR2HSV
	))

	y, cr, cb = cv2.split(imageYCC)

	convolutionYCC = np.zeros([height, width, 1], dtype = np.uint8)

	yccSpectrum = [[0 for z in range(16)] for i in range(16)]

	for x in range(width):
		for y in range(height):
			yccSpectrum[round(imageYCC.item(y, x, 1) / 16)][round(imageYCC.item(y, x, 2) / 16)] += 1

	maxDimension = [0, 0]
	maxValue = 0
	prevMax = 0
	prevMaxDim = [0, 0]

	o = 0
	for line in yccSpectrum:
		m = 0
		for point in line:
			if point > maxValue:
				prevMax = maxValue
				prevMaxDim = maxDimension
				maxDimension = [o, m]
				maxValue = point
			m += 1
		o += 1

	for x in range(width):
		for y in range(height):
			colorValue1 = round(imageYCC.item(y, x, 1) / 16)
			colorValue2 = round(imageYCC.item(y, x, 2) / 16)

			if (maxDimension[0] == colorValue1 and maxDimension[1] == colorValue2) or (prevMaxDim[0] == colorValue1 and prevMaxDim[1] == colorValue2):
				convolutionYCC.itemset((y, x, 0), 255)

	"""
	dataYCC = np.float32(
		convolutionYCC.reshape((-1, 1))
	)

	colorBins = plt.hist(dataYCC, 256, [0, 256])[0]

	maxMinColors = getLocalMaxMin(colorBins)
	centerColor = maxMinColors[0][
		getClosestVal(maxMinColors[0], 124)
	]
	colorRadius = abs(
		centerColor - maxMinColors[1][
			getClosestVal(maxMinColors[1], centerColor)
		]
	)

	plt.close('all')

	return (centerColor, colorRadius, convolutionYCC)
	

	for x in range(width):
		for y in range(height):
			convolutionYCC.itemset((y, x, 0), convolutionYCC.item(y, x, 0) - )
	"""

	#print(yccSpectrum)
	
	for x in range(width):
		for y in range(height):
			calcVal = (int(a[y, x]) + int(s[y, x])) / 2
			begVal = convolutionYCC.item(y, x, 0)

			if calcVal < 110 and calcVal > 50 and begVal == 255:
				convolutionYCC.itemset((y, x, 0), calcVal ** (1 / 3))
			elif calcVal < 110 and calcVal > 50:
				convolutionYCC.itemset((y, x, 0), calcVal ** (1 / 2))
			else:
				convolutionYCC.itemset((y, x, 0), calcVal)
	
	cd = cv2.Canny(convolutionYCC, 450, 530, apertureSize = 3)

	#topLineCannyDetection = cv2.Canny(s, 350, 430, apertureSize = 3)
	lines = cv2.HoughLinesP(cd, rho = 1, theta = 1/360, threshold = 200, minLineLength = 200,maxLineGap = 0)
	"""
	if type(lines) != type(None):
		for i in range(2):
			r = lines[i][0][0]
			m = lines[i][0][1]
			drawLine(cd, r, m, 150, 1)
	"""
	if type(lines) != type(None):
		N = lines.shape[0]
		for i in range(N):
			x1 = lines[i][0][0]
			y1 = lines[i][0][1]    
			x2 = lines[i][0][2]
			y2 = lines[i][0][3]    
			cv2.line(image,(x1,y1),(x2,y2),(0, 255, 0),1)



	displayImage("im", image)
	#displayImage("im", topLineCannyDetection)

def experiment2():
	inputVideo = cv2.VideoCapture("illiniClip.mp4")
	framesPerRead = 10
	firstRead = True
	frameIndex = 1
	prevFrame = None
	centerX = 0
	centerY = 0
	while inputVideo.isOpened() and frameIndex < 900:
		start = time.time()
		frame = inputVideo.read()[1]
		prevFrame = frame

		if firstRead:
			centerX = frame.shape[1] / 2
			centerY = frame.shape[0] / 2
			for i in range(30):
				frame = inputVideo.read()[1]
				frameIndex += 1
			frame2 = frame[(int(frame.shape[0] * 0.375)):(int(frame.shape[0] * 0.8)), 0:frame.shape[1]]
			#video = cv2.VideoWriter('goodLineDetection2.avi', -1, 1, (frame2.shape[1], frame2.shape[0]))
			#print(frame.shape)
			prevFrame = frame
			firstRead = False

		for i in range(framesPerRead - 1):
			frame = inputVideo.read()[1]
			frameIndex += 1

		offset = int(frame.shape[0] * 0.375)

		imageResult, detection = analyzeDetection(frame, crop = True)
		tl = detection[0]
		height, width = frame.shape[0:2]
		#print(frame.shape)
		endVals = getPolarEndpoints(tl[0], tl[1])
		topLineMask = np.zeros([height, width], dtype = np.uint8)
		cv2.line(topLineMask, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 30)
		sChannelFromHSVImage = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[1]
		topLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 38000, 50000, apertureSize = 7)
		topLineCannyDetection[:] = topLineCannyDetection[:] / 2
		#topLineCannyDetection[topLineCannyDetection == 255] = 100
		#tl2 = topLineCannyDetection.copy()
		topLineCannyDetection[topLineMask < 255] = 0
		intersections = np.zeros([height, width], dtype = np.uint8)
		cv2.line(intersections, (endVals[0][0], endVals[0][1] + offset), (endVals[1][0], endVals[1][1] + offset), 255, 1)
		extraMask = np.zeros([height, width], dtype = np.uint8)

		
		lines = cv2.HoughLines(topLineCannyDetection,1,np.pi/180,20)
		newLines = []
		for line in reversed(lines):
			for rho,theta in line:
				if not (abs(tl[1] - theta) > math.radians(25)):
					continue
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))

				#cv2.line(topLineCannyDetection,(x1,y1),(x2,y2),rank,1)
				newLines.append([(x1,y1),(x2,y2)])
		rank = 255 - len(newLines) * 51
		for line in newLines:
			rank += 64
			if rank > 0:
				cv2.line(extraMask,line[0],line[1],rank,1)
		extraMask[intersections == 0] = 0

		circling = False
		circleInProgressX = 0
		circleInProgressY = 0
		circleInProgressWeight = 0
		circleVals = 0
		circles = []
		dist = 0
		for x in range(width):
			if circling:
				dist += 1
			for y in range(height):
				if extraMask.item(y,x) != 0:
					if circling:
						if dist >= 20:
							circles.append([(int(circleInProgressX / (circleInProgressWeight)), 
							int(circleInProgressY / (circleInProgressWeight))), circleInProgressWeight / circleVals])
							circling = False
							circleInProgressX = 0
							circleInProgressY = 0
							circleInProgressWeight = 0
							circleVals = 0
							dist = 0
							break
						cValue = extraMask.item(y, x)
						circleInProgressX += x * cValue
						circleInProgressY += y * cValue
						circleInProgressWeight += cValue
						circleVals += 1
					else:
						circling = True
						cValue = extraMask.item(y, x)
						circleInProgressX += x * cValue
						circleInProgressY += y * cValue
						circleInProgressWeight += cValue
						circleVals += 1
					break

					#cv2.circle(frame,(x,y),int(5 * (extraMask.item(y,x) / 255)),(255, 50, 255),-1)
		if circleVals != 0:
			circles.append([(int(circleInProgressX / (circleInProgressWeight)), 
							int(circleInProgressY / (circleInProgressWeight))), circleInProgressWeight / circleVals])

		for circle in circles:
			cv2.circle(frame, circle[0], 2, (140, 0, 255), -1)
		#for x in range(width):
		#	for y in range(height):
		#		if tl2.item(y, x) != 0 and y > 0 and y < height - 1:
		#			if tl2.item(y - 1, x) == 0 and tl2.item(y + 1, x) == 0:
		#				tl2.itemset((y, x), 255)

		#nf = detectLineElements(tl2)
		#topLineCannyDetection[nf == 255] = 255
		MIN_MATCH_COUNT = 1

		img1 = prevFrame[int(max(0, centerY - 70)):int(min(frame.shape[1], centerY + 70)), 0:frame.shape[1]]          # queryImage
		img2 = frame[int(max(0, centerY - 70)):int(min(frame.shape[1], centerY + 70)), 0:frame.shape[1]] # trainImage

		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 500)

		flann = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

		matches = flann.match(des1, des2)
		matches = sorted(matches, key = lambda x:x.distance)
		mats = {}
		distances = []

		for mat in matches:
			# Get the matching keypoints for each of the images
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

			# x - columns
			# y - rows
			# Get the coordinates
			(x1,y1) = kp1[img1_idx].pt
			(x2,y2) = kp2[img2_idx].pt
			di = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1/2))
			if di < 800:
				if di not in mats:
					mats[di] = []
				mats[di].append((x2 - x1, y2 - y1))
			distances.append(di)
		mode = max(set(distances), key = distances.count)
		#print(dx / len(mats), dy / len(mats))
		dx = 0
		dy = 0
		for uy in mats[mode]:
			dx += uy[0]
			dy += uy[1]
		dx /= len(mats[mode])
		dy /= len(mats[mode])
		centerX += dx
		centerY += dy
		

		"""
		# store all the good matches as per Lowe's ratio test.
		N_MATCHES = 40

		match_img = cv2.drawMatches(
			img1, kp1,
			img2, kp2,
			matches[:N_MATCHES], img2.copy(), flags=0)
		"""
		#for uu in mats:
		#	cv2.circle(frame, uu, 2, (255, 40, 40), -1)
		cv2.circle(frame, (int(centerX), int(centerY)), 7, (40, 255, 40), -1)

		#print(time.time() - start)
		frameIndex += 1
		winname = "Test"
		cv2.namedWindow(winname)        # Create a named window
		cv2.moveWindow(winname, 40,30)
		cv2.imshow(winname, frame)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		else:
			cv2.destroyAllWindows()



#imgReg = cv2.imread("gameShot copy 3.png", 1)
#experiment2()


"""

cap = cv2.VideoCapture('illiniClip.mp4')
ret2, frame2 = cap.read()
frame2 = frame2[(int(frame2.shape[0] * 0.375)):(int(frame2.shape[0] * 0.8)), 0:frame2.shape[1]]
video = cv2.VideoWriter('topLineDetection.mp4', -1, 1, (frame2.shape[1], frame2.shape[0]))

gameImage = None
uuuu = False
while(cap.isOpened()):
	ret, frame = cap.read()
	if uuuu:
		for i in range(5000):
			ret, frame = cap.read()
		uuuu = False

	for i in range(2):
		ret, frame = cap.read()

	#courtEdges = findCourtInShot(frame)
	

	 
	#frame = frame[(int(frame.shape[0] * 0.375)):(int(frame.shape[0] * 0.8)), 0:frame.shape[1]]
	edges = detectCourtEdges(frame, displayResult = False, shouldCrop = True)
	video.write(edges)
	#gameImage = frame
	cv2.imshow('frame', edges)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video.release()
#cv2.imwrite('gameShot.png', gameImage)

"""
