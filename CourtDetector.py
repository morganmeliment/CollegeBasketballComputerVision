# Abstract class for a court detection method.
# 
#	Author ---- Created by Morgan Meliment on May 18th 2018.
#	Github ---- github.com/morganmeliment
#	LinkedIn -- linkedin.com/in/morganmeliment
#	Email ----- morganm4@illinois.edu

class CourtDetector:
	video = None
	height = 0
	width = 0

	# Keeps track of detections for reconciling conflicting information.
	analyzedFrames = {}

	def __init__(self, video):
		self.video = video
		self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

	

