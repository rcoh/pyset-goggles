from SimpleCV import *
import time
import math
from collections import Counter

def identify_cards(filename):
	img = Image('and1.jpg').resize(800)
	blobs = img.findBlobs()
	print blobs[-12:]
	blob_colors = [(blob, identify_card(blob)) for blob in blobs[-9:]]
	index = 0
	disp = Display()

	while disp.isNotDone():
	        if disp.mouseLeft:
	                index += 1           
	        if disp.mouseRight:
	        		break
	        (blob, color) = blob_colors[index]
	        print color
	        blob.hullImage().save(disp)	
	        time.sleep(1)


def identify_card(card_blob):
	ret = (identify_number(card_blob), identify_color(card_blob), identify_shape(card_blob))
	disp = Display()
	print ret
	while disp.isNotDone():
		 if disp.mouseLeft:
		 	break
		 nice = card_blob.hullImage()
		 mask = card_blob.hullImage().colorDistance(Color.WHITE).binarize(100)
		 onlyShapes = card_blob.hullImage() - mask
		 pointsLayer = DrawingLayer((onlyShapes.width, onlyShapes.height))
		 points = onlyShapes.findBlobs()[-1].contour()
		 i = 0
		 for point in points[0::2]:
		 	pointsLayer.circle(point, i, color = Color.RED)

		 nice.addDrawingLayer(pointsLayer)
		 nice.applyLayers()
		 nice.save(disp)
	
	return ret

def identify_number(card_blob):
	mask = card_blob.hullImage().colorDistance(Color.WHITE).binarize(100)
	onlyShapes = card_blob.hullImage() - mask
	return len(onlyShapes.findBlobs()) 

def identify_shape(card_blob):
	mask = card_blob.hullImage().colorDistance(Color.WHITE).binarize(100)
	onlyShapes = card_blob.hullImage() - mask
	hull = onlyShapes.findBlobs()[-1].contour()
	shape_descriptor = get_shape_descriptor(hull)
	print shape_descriptor
	if shape_descriptor[0.0] > .90:
		return "diamond"
	elif shape_descriptor[0.0] > .83:
		return "oval"
	else:
		return "squiggle"


def get_shape_descriptor(points):
	descriptor = Counter()
	bucket_size = math.pi / 20
	points = points[0::2]
	for i in range(len(points) - 2):
		(v1, v2, v3) = points[i:i+3]
		try:
			angle = get_angle(v1, v2, v3)
			descriptor[math.floor(angle / bucket_size) * bucket_size] += 1
		except:
			print "error:", v1, v2, v3

	total = sum(descriptor.values())
	for ang in descriptor:
		descriptor[ang] /= (total + 0.0)
	return descriptor

def get_angle(v1, v2, v3):
	p12 = dist(v1, v2)
	p13 = dist(v1, v3)
	p23 = dist(v2, v3)
	if v1 == v2 or v1 == v3:
		return -1
	return math.acos((p12 ** 2 + p13 ** 2 - p23 ** 2) / (2 * p12 * p13))	

def dist(v1, v2):
	(x1, y1) = v1
	(x2, y2) = v2
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def identify_color(card_blob):
	mask = card_blob.hullImage().colorDistance(Color.WHITE).binarize(100)
	onlyShapes = card_blob.hullImage() - mask
	(b,g,r) = onlyShapes.meanColor()

	if r > g and r > b:
		color = "red"
	if g > r and g > b:
		color = "green"
	if b > r and b > g:
		color = "purple"

	return color

identify_cards("ex1.jpg")


