# Class that expands functionality of an OpenCV image.
# 
#	Author ---- Created by Morgan Meliment on May 18th 2018.
#	Github ---- github.com/morganmeliment
#	LinkedIn -- linkedin.com/in/morganmeliment
#	Email ----- morganm4@illinois.edu

class SmartFrame():
	frame = None
	height = 0
	width = 0

	def __init__(self, frame):
		self.frame = frame
		self.height = frame.shape[0]
		self.width = frame.shape[1]

	#
	# Create a histogram of the court colors in the frame.
	#
	#	@return -- The center color of the frame, the radius of valid court colors, and the
	# 				flattened image.
	#
	def getCourtColors(self):
		# Convert image from RGB to YCC colorspace.
		imageYCC = cv2.cvtColor(
			self.frame,
			cv2.COLOR_BGR2YCR_CB
		)

		# Flatten YCC image to one dimension.
		convolutionYCC = np.zeros([self.height, self.width, 1], dtype = np.uint8)
		for x in range(self.width):
			for y in range(self.height):
				convolutionYCC.itemset(
					(y, x, 0),
					int((imageYCC.item(y, x, 1) + imageYCC.item(y, x, 2)) / 2)
				)
		
		# Flatten into a list of pixels.
		dataYCC = np.float32(
			convolutionYCC.reshape((-1, 1))
		)

		# Create a histogram of the pixel values.
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

		# Return the center color of the image, the radius of valid court colors, and the
		# flattened image.
		return (centerColor, colorRadius, convolutionYCC)

	#
	# Analyze court-like pixels.
	#
	#	@return -- The image containing binary pixels (1 = Court Edge Like, 0 = Unlikely)
	#
	def getCourtEdges(self):
		centerColor, colorRadius, convolutionYCC = self.getCourtColors()
		courtEdges = np.zeros([self.height, self.width, 1], dtype = np.uint8)
		pixelColumn = 0
		pixelRow = 0

		# If radius = 1, we only need to check for one court color.
		if colorRadius == 1:
			# Check each pixel for court like-ness.
			while pixelColumn < self.width:
				convolutionItem = convolutionYCC.item(pixelRow, pixelColumn, 0)

				if (abs(convolutionItem - centerColor) < colorRadius
						or convolutionItem == centerColor - colorRadius - 1):
					matches = 0

					maxRow = 10
					if maxRow + pixelRow > self.height:
						maxRow = self.height - pixelRow
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
					if pixelRow >= self.height:
						pixelRow = 0
						pixelColumn += 1
		else:
			# Check each pixel for court like-ness.
			while pixelColumn < self.width:
				if abs(convolutionYCC.item(pixelRow, pixelColumn, 0) - centerColor) < colorRadius:
					matches = 0

					maxRow = 10
					if maxRow + pixelRow > self.height:
						maxRow = self.height - pixelRow
					for i in range(1, maxRow):
						if abs(convolutionYCC.item(pixelRow + i, pixelColumn, 0) - centerColor) < colorRadius:
							matches += 1

					if matches >= 3:
						courtEdges.itemset((pixelRow, pixelColumn, 0), 220)

					pixelColumn += 1
					pixelRow = 0
				else:
					pixelRow += 1
					if pixelRow >= self.height:
						pixelRow = 0
						pixelColumn += 1

		# Return the binary image of court pixels.
		return courtEdges

	#
	# Detect and return the polar equation of the top line of the court.
	#
	#	@return -- The polar equation of the top line.
	#
	def getTopLine(self):
		# Save initial analyzations of the image to process later.
		courtEdges = self.getCourtEdges()
		sChannelFromHSVImage = cv2.split(cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV))[1]
		lines = cv2.HoughLines(courtEdges, 5, np.pi/180, 30)
		topLineCannyDetection = cv2.Canny(sChannelFromHSVImage, 350, 430, apertureSize = 3)
		topLineMask = np.zeros([self.height, self.width, 1], dtype = np.uint8)

		# Keep a running total of the detections to average later.
		mainR = 0
		mainTheta = 0
		linesIncluded = 0

		# Isolate the general area where the top line is likely to be.
		if type(lines) != type(None):
			drawLine(topLineMask, lines[0][0][0], lines[0][0][1], 255, 30)

		# Erase area that is unlikely to contain the top line.
		for x in range(self.width):
			for y in range(self.height):
				if topLineMask.item(y, x, 0) != 255:
					topLineCannyDetection.itemset((y, x), 0)
		
		# Run a new line detection on the area that is likely to contain the top line.
		lines = cv2.HoughLines(topLineCannyDetection, 5, np.pi/180, 30)

		# Average the top 2 (or fewer) results.
		if type(lines) != type(None):
			for i in range(2):
				mainR += lines[i][0][0]
				mainTheta += lines[i][0][1]
				linesIncluded += 1

		# Return the averaged result if one is found.
		if linesIncluded != 0:
			return (mainR / linesIncluded, mainTheta / linesIncluded)
		else:
			return (0, 0)

	



















