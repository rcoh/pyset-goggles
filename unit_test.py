import os
import envoy
test_files = os.listdir("ground_truth")
for test_file in test_files:
	file_name = os.path.basename(test_file)
	test_name = os.path.splitext(file_name)[0]
	input_image = "training/%s.jpg" % test_name
	if not os.path.exists(input_image):
		print "WARN: %s does not exist" % input_image
		continue
	print "Running", file_name
	with file("test_output/%s" % file_name, "w") as outfile:
		res = envoy.run('python analyzer.py -f %s' % (input_image))
		if res.status_code != 0:
			print "Status != 0"
			print res.std_err
			raise
		outfile.write(res.std_out)

	diff = envoy.run('diff ground_truth/%s test_output/%s' % (file_name, file_name))
	if diff.status_code != 0:
		print "Output does not match! %s vs. %s" % (test_file, test_name)
		print diff.std_out
		raise

print "Success!"

