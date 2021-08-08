# Extra Code



def load_image(img_path, shape=None):
    img = cv2.imread(img_path)
    
    return img

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return deg * 180.0 / pi

class ImageTransformer(object):

    def __init__(self, image):
        self.image = image;
 
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        
        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

def getColorDistance(color1, color2):
	return (((color1[0] - color2[0]) ** 2) + ((color1[1] - color2[1]) ** 2) + ((color1[2] - color2[2]) ** 2)) ** (1/2)








"""
courtImage = cv2.imread("gameShot.png", 0)
courtEdges = cv2.Canny(courtImage, 100, 200)
#transformer = ImageTransformer(courtEdges)


#while True:

#timage = transformer.rotate_along_axis(theta = 30)

cv2.imshow("Court Image", courtEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()

def pullEdgesFromImage(res2, K):
    newColArray = []
    width = res2.shape[1]
    height = res2.shape[0]

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)
        mult = int(((y - height/2)*(3.3/height)) ** 6)
        for j in range(mult):
            newColArray.append(res2[y, x])

    newColDiff = (len(newColArray) - int(len(newColArray) ** (1/2)) ** 2) * -1
    newColArray = newColArray[0:newColDiff]

    Z = np.float32(newColArray)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res3 = res.reshape(int(len(newColArray) ** (1/2)), int(len(newColArray) ** (1/2)), 3)
    return res3

def getCourtLineColor(courtImage):
    courtEdges = cv2.Canny(courtImage, 200, 250, apertureSize = 3)
    width = courtImage.shape[1]
    height = courtImage.shape[0]
    newImage = courtImage.copy()
    minLineLength = 400
    maxLineGap = 10
    lines = cv2.HoughLinesP(courtEdges, 1, np.pi / 180, int(height * 0.7), None, 300, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(newImage, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, cv2.LINE_AA)

    colorArray = {}
    colArray = []

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if newImage.item(y, x, 1) == 255:
            colArray.append(courtImage[y, x])

    colDiff = (len(colArray) - int(len(colArray) ** (1/2)) ** 2) * -1
    colArray = colArray[0:colDiff]

    Z = np.float32(colArray)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(int(len(colArray) ** (1/2)), int(len(colArray) ** (1/2)), 3)

    res2 = pullEdgesFromImage(res2, 12)
    res2 = pullEdgesFromImage(res2, 2)

    color1 = res2[0, 0]
    color2 = res2[0, 1]
    width = res2.shape[1]
    height = res2.shape[0]
    total = 2
    while (total < width * height and color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]):
        x = total % width
        y = math.floor((total - x) / width)
        color2 = res2[y, x]
        total += 1

    width = 300
    height = 50
    pickerImage = np.zeros((height,width,3), np.uint8)
    pickerImage[:,0:int(0.5*width)] = color1
    pickerImage[:,int(0.5*width):width] = color2

    while (True):
        cv2.imshow('Press 0 for the left color, 1 for right', pickerImage)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            return color2
            break
        elif cv2.waitKey(1) & 0xFF == ord('0'):
            return color1
            break
"""


def isLeftEnd(angle):
    degreeAngle = (angle * 180 / math.pi) + 90
    print(degreeAngle)

def getDominantColors(imgReg):
    imgYCC = cv2.cvtColor(imgReg, cv2.COLOR_BGR2YCR_CB)
    bbbb,gb,rb = cv2.split(imgYCC)
    imgYCC = gb
    width = imgYCC.shape[1]
    height = imgYCC.shape[0]

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)
        imgYCC[y, x] = int(gb[y, x] / 2) + int(rb[y, x] / 2)

    courtImage = imgYCC.copy()
    courtEdges = imgYCC.copy() #cv2.Canny(imgReg, 450, 530, apertureSize = 3)
    courtEdges2 = imgYCC.copy()
    Z = courtImage.reshape((-1,1))
    Z = np.float32(Z)
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((courtImage.shape))

    res2 = courtImage.copy()"""

    f, g, p = plt.hist(Z,256,[0,256])
    #plt.show()

    maxMin = getLocalMaxMin(f)
    centerCol = maxMin[0][getClosestVal(maxMin[0], 124)]
    radius = abs(centerCol - maxMin[1][getClosestVal(maxMin[1], centerCol)])
    #print(maxMin, centerCol, radius)

    width = courtImage.shape[1]
    height = courtImage.shape[0]
    maxPoints = {}

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if x not in maxPoints:
            maxPoints[x] = 0

        courtEdges2[y, x] = 0
        courtEdges[y, x] = 0
        if radius == 1:
            if abs(courtImage[y, x] - centerCol) < radius or courtImage[y, x] == centerCol - radius - 1:
                courtEdges2[y, x] = 100
                if maxPoints[x] == 0:
                    maxPoints[x] = y
                    if x != 0 and abs(maxPoints[x - 1] - maxPoints[x]) < 2:
                        courtEdges[y, x] = 220
                        imgReg[y, x] = (0, 150, 0)
        else:
            if abs(courtImage[y, x] - centerCol) < radius:
                courtEdges2[y, x] = 100
                if maxPoints[x] == 0:
                    maxPoints[x] = y
                    if x != 0 and abs(maxPoints[x - 1] - maxPoints[x]) < 2:
                        courtEdges[y, x] = 220
                        imgReg[y, x] = (0, 150, 0)

    """
    colorDict = {}
    width = courtImage.shape[1]
    height = courtImage.shape[0]
    mainColor = 0
    mainColorData = [0, 0]

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if str(res2[y, x]) in colorDict:
            colorDict[str(res2[y, x])] += 1
            if colorDict[str(res2[y, x])] > mainColor:
                mainColor = colorDict[str(res2[y, x])]
                mainColorData = res2[y, x]
        else:
            colorDict[str(res2[y, x])] = 1

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if courtImage[y, x][0] == mainColorData[0] and courtImage[y, x][1] == mainColorData[1]:
            imgReg[y, x] = (255, 255, 255)
    #mainColor = [mainColor.split("")]"""

    lines = cv2.HoughLines(courtEdges,5,np.pi/180, 30)
    if type(lines) != type(None):
        horizontalTheta = 0
        for r,theta in lines[0]:
            #isLeftEnd(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1600*(-b))
            y1 = int(y0 + 1600*(a))
            x2 = int(x0 - 1600*(-b))
            y2 = int(y0 - 1600*(a))
            cv2.line(courtEdges2,(x1,y1), (x2,y2), 255, 2)
            horizontalTheta = theta + math.pi / 2
            if horizontalTheta % math.pi != horizontalTheta:
                r = -1 * r
                theta = horizontalTheta % math.pi
            else:
                theta = horizontalTheta
            #print(np.sin(theta) * (height / np.cos(theta)))
            theta += math.radians(52)
            nMult = np.sin(theta) * (height / np.cos(theta))
            gMult = 0
            if nMult < 0:
                gMult = nMult
                nMult = -1 * nMult

            m = (y2 - y1) / (x2 - x1)
            mx1 = (-1 * m * x1) + y1

            densityDict = {}


            for i in range(int((width + nMult) / 3)):
                lineHeight = height - int(m*(3*i + int(gMult)) + mx1)
                a = np.cos(theta)
                b = np.sin(theta)
                x1 = i * 3 + int(gMult)
                y1 = int(m*(3*i + int(gMult)) + mx1)
                x2 = int(i * 3 + int(gMult) - (lineHeight / a) * b)
                y2 = height
                cv2.line(courtEdges2,(x1, y1), (x2, y2), 255, 1)

                xDiff = x2 - x1
                yDiff = y2 - y1
                hypotenuse = int(((xDiff ** 2) + (yDiff ** 2)) ** (1 / 2))

                density = 0

                prevXN = -90000
                prevYN = -90000

                for j in range(hypotenuse):
                    xN = int(i * 3 + int(gMult) + b * j)
                    yN = int(m*(3*i + int(gMult)) + mx1 - a * j)

                    if xN != prevXN or yN != prevYN:
                        prevXN = xN
                        prevYN = yN

                        try:
                            if courtEdges2[yN, xN] == 100:
                                density += 1
                        except:
                            pass

                densityDict[x1] = density

            plt.clf()
            plt.cla()
            plt.close()

            listst = sorted(densityDict.items()) # sorted by key, return a list of tuples

            xttt, yttt = zip(*listst) # unpack a list of pairs into two tuples

            plt.plot(xttt, yttt)
            plt.show()
            #print(densityDict)

    """
    dst = cv2.cornerHarris(courtEdges2,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    #Threshold for an optimal value, it may vary depending on the image.
    courtEdges2[dst>0.95*dst.max()]=255
    
    #return imgReg
    """
    cv2.imshow("Court Image", courtEdges2)
    cv2.waitKey(0)
    #return 0
    cv2.destroyAllWindows()


"""imgYCC = cv2.cvtColor(imgReg, cv2.COLOR_BGR2YCR_CB)
print(imgYCC[0,0])
bbbb,gb,rb = cv2.split(imgYCC)
imgYCC = gb
width = imgYCC.shape[1]
height = imgYCC.shape[0]
for i in range(width * height):
    x = i % width
    y = math.floor((i - x) / width)
    imgYCC[y, x] = int(gb[y, x] / 2) + int(rb[y, x] / 2)
"""
#print(imgYCC.shape)

#imgYCC = cv2.Canny(imgYCC, 10, 20)
#getDominantColors(imgReg)

"""
imgReg = cv2.imread("gameShot copy.png", -1)
detectCourtEdges(imgReg)
imgReg = cv2.imread("gameShot copy 3.png", -1)
detectCourtEdges(imgReg)
imgReg = cv2.imread("gameShot copy 4.png", -1)
detectCourtEdges(imgReg)
"""
#getDominantColors(rb, bbbb, imgReg)
#getDominantColors(gb, bbbb, imgReg)
#print("Illinois: " + str(getCourtLineColor(cv2.imread("ilCourt.jpg", 1))))
#getCourtLineColor(cv2.imread("nwCourt.jpg", -1), 0.1)
#getCourtLineColor(cv2.imread("ilCourt.jpg", -1), 0.1)
"""
def getSurroundingPixel(n):
    if n > 8:
        n = 8
    elif n < 0:
        n = 0
    d = "r"
    w = -1
    h = 0
    for u in range(n + 1):
        if d == "r":
            if w < 2:
                w += 1
            else:
                d = "d"
                h += 1
        elif d == "d":
            if h < 2:
                h += 1
            else:
                d = "l"
                w -= 1
        elif d == "l":
            if w > 0:
                w -= 1
            else:
                d = "u"
                h -= 1
    return [w, h]

def findCourtInShot(gameImage):
    courtEdges = cv2.Canny(gameImage, 100, 450, apertureSize = 3)
    width = gameImage.shape[1]
    height = gameImage.shape[0]
    newImage = courtEdges.copy()
    newerImage = courtEdges.copy()
    newestImage = courtEdges.copy()
    newestImage[:] = 0

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if courtEdges[y, x] == 255:
            weights = [0.5, -2.0, 0.5, 2.0, 0.5, -2.0, 0.5]
            matchWeights = []
            pixelMask = []

            for u in range(8):
                offSet = getSurroundingPixel(u)
                newX = x + offSet[0] - 1
                newY = y + offSet[1] - 1
                on = False

                if not (newX < 0 or newX >= width or newY < 0 or newY >= height):
                    on = (courtEdges[newY, newX] == 255)

                if on:
                    on = 1.0
                else:
                    on = 0.0

                pixelMask.append(on)

            n = 0
            for u in pixelMask:
                weight = 0
                if u == 1.0:
                    ny = 0
                    o = 0
                    for h in pixelMask:
                        if ny > n:
                            weight += h * weights[o]
                            o += 1
                        ny += 1
                    ny = 0
                    for h in pixelMask:
                        if ny < n:
                            weight += h * weights[o]
                            o += 1
                        ny += 1
                n += 1

                matchWeights.append(weight)

            pixelVal = 0

            if sum(matchWeights) > 3.9:
                pixelVal = 255 #int((sum(matchWeights) + 4.0) * 31)

            newImage[y, x] = pixelVal
            #print(str(newImage[y, x]) + " y: " + str(y) + " x: " + str(x))

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        if newImage[y, x] == 255:
            weights = [0.5, -2.0, 0.5, 2.0, 0.5, -2.0, 0.5]
            matchWeights = []
            pixelMask = []

            for u in range(8):
                offSet = getSurroundingPixel(u)
                newX = x + offSet[0] - 1
                newY = y + offSet[1] - 1
                on = False

                if not (newX < 0 or newX >= width or newY < 0 or newY >= height):
                    on = (newImage[newY, newX] == 255)

                if on:
                    on = 1.0
                else:
                    on = 0.0

                pixelMask.append(on)

            n = 0
            for u in pixelMask:
                weight = 0
                if u == 1.0:
                    ny = 0
                    o = 0
                    for h in pixelMask:
                        if ny > n:
                            weight += h * weights[o]
                            o += 1
                        ny += 1
                    ny = 0
                    for h in pixelMask:
                        if ny < n:
                            weight += h * weights[o]
                            o += 1
                        ny += 1
                n += 1

                matchWeights.append(weight)

            pixelVal = 0

            if sum(matchWeights) > -5:
                pixelVal = int((sum(matchWeights) + 4.0) * 31)

            newerImage[y, x] = pixelVal
            #print(str(newImage[y, x]) + " y: " + str(y) + " x: " + str(x))
        else:
            newerImage[y, x] = 0

    for i in range(width * height):
        x = i % width
        y = math.floor((i - x) / width)

        try:
            if (newerImage[y, x] == 124 or newerImage[y, x] == 155) and newerImage[y, x - 1] == 0 and newerImage[y, x + 1] == 248:
                h = 1
                while newerImage[y, x + h] == 248:
                    h += 1
                if (newerImage[y, x + h] == 124 or newerImage[y, x + h] == 155) and h > 7:
                    for u in range(h + 2):
                        newestImage[y, x + u] = 255
        except:
            pass

    return newestImage
""""""
illnoisRGB = [53, 34, 26]
illinoisBW = sum(illnoisRGB) / 3

courtEdges = findCourtInShot(cv2.imread("gameShot.png", 1))
cv2.imshow("Court Image", courtEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()

courtEdges = cv2.Canny(cv2.imread("gameShot.png", 0), 100, 450, apertureSize = 3)
cv2.imshow("Court Image", courtEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
image = cv2.imread("gameShot.png", 0)
courtEdges = cv2.Canny(image, 100, 450, apertureSize = 3)
cv2.imshow("Court Image", courtEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()
width = image.shape[1]
height = image.shape[0]

for i in range(width * height):
    x = i % width
    y = math.floor((i - x) / width)

    if image[y, x] <= 65:
        image[y, x] += 10

"""
