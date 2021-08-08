# Collection of helper functions for court detection.
# 
#	Author ---- Created by Morgan Meliment on May 12th 2018.
#	Github ---- github.com/morganmeliment
#	LinkedIn -- linkedin.com/in/morganmeliment
#	Email ----- morganm4@illinois.edu

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from memory_profiler import profile


#
# Get the endpoints of a line from the cv2.HoughLines function.
#
# 	@param bins    The color bins to analyze.
#
#	@return A list containing a list for each type of endpoint.
#
def getLocalMaxMin(bins):
	prevVal = 0
	maxima = []
	minima = []

	# Detect each local min/max by analyzing each point.
	for i in range(len(bins)):
		nextVal = bins[i]
		if i != len(bins) - 1:
			nextVal = bins[i + 1]

		if prevVal < bins[i]:
			# Found max critical point
			if nextVal < bins[i]:
				maxima.append(i)
		else:
			# Found min critical point
			if prevVal > bins[i] and nextVal > bins[i]:
				minima.append(i)

		# Keep track of the previous value.
		prevVal = bins[i]

	# Return each list.
	return [maxima, minima]

#
# Find the closest match to the given value.
#
# 	@param arr    The list (array) of values to check.
# 	@param val    The value to check for.
#
#	@return The index of the closest match.
#
def getClosestVal(arr, val):
	if len(arr) == 0:
		return 0

	minDist = abs(val - arr[0])
	minIndex = 0

	for i in range(1, len(arr)):
		# Check each value's distance from val.
		if abs(val - arr[i]) < minDist:
			minDist = abs(val - arr[i])
			minIndex = i

	# Return the index of the closest match.
	return minIndex

#
# Get the endpoints of a line from the cv2.HoughLines function.
#
# 	@param r    The given "r" value by cv2.HoughLines.
# 	@param theta    The given "theta" value by cv2.HoughLines.
# 	@param width    The width of the image.
# 	@param height    The height of the image.
#
#	@return A list containing an object for each endpoint.
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

	# A list containing each endpoint.
	return [firstPoint, secondPoint]

#
# Evaluate a polar function at a cartesian x-coordinate.
#
# 	@param r    The given "r" value by cv2.HoughLines.
# 	@param theta    The given "theta" value by cv2.HoughLines.
# 	@param x    The desired x coordinate.
#
#	@return The function value (y) at the given x-coordinate.
#
def functionOfX(r, theta, x):
	a, b = np.cos(theta), np.sin(theta)
	x0, y0 = a * r, b * r

	# Calculate two points on the function.
	x1 = int(x0 - 1600 * b)
	y1 = int(y0 + 1600 * a)
	x2 = int(x0 + 1600 * b)
	y2 = int(y0 - 1600 * a)

	# Calculate the slope given two points.
	# + Protects against division by 0.
	m = 100000 if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)

	# Return y-value at given x-value of this function.
	return int(y1 + m * (x - x1))

# Court Area From Image (Seems to be unused currently)

#
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

#
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

#
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

#
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

#
# Calculate the cartesian point of intersection between two polar equations.
#
#	@param r1 ------- "R" value for first line eq.
#	@param theta1 --- "Theta" value for first line eq.
# 	@param r2 ------- "R" value for second line eq.
#	@param theta2 --- "Theta" value for second line eq.
#
#	@return --------- The cartesian point of intersection.
#
def getLineIntersection(r1, theta1, r2, theta2):
	cartesianVals1 = polarToCartesian(r1, theta1)
	cartesianVals2 = polarToCartesian(r2, theta2)

	m1, b1 = cartesianVals1["m"], cartesianVals1["b"]
	m2, b2 = cartesianVals2["m"], cartesianVals2["b"]

	# Check for division by zero, then calculate the result.
	x = -100
	if m1 - m2 != 0:
		x = int((b2 - b1) / (m1 - m2))
	y = int(m1 * x + b1)

	# Return the point of intersection.
	return (x, y)

#
# Simplified OpenCV display image helper.
#
#	@param title ---- The title of the image window.
#	@param image ---- The image to display.
#
def displayImage(title, image):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()









































































