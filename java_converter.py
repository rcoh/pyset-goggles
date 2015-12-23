import pickle
import sys

if __name__ == "__main__":
	path = sys.argv[1]
	class_name = sys.argv[2]
	with file(path) as f:
		contour = pickle.load(f)
	points, x, dims = contour.shape
	assert x == 1
	assert dims == 2
	point_strs = []
	for i in range(points):
		x,y = contour[i][0]
		point_strs.append("new Point(%s, %s)" % (x,y))
	args = ', '.join(point_strs)
	java_file = """
		class %s {
			static MatOfPoint contour = new MatOfPoint(%s);
		}
	""" % (class_name, args)

	with file(sys.argv[3], 'w') as f:
		f.write(java_file)
