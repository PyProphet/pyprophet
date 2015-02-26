# encoding: latin-1

def nice_time(t):
	seconds = int(t)
	msecs 	= int(1000 * (t - seconds))
	minutes = int(t / 60.0)
	hours 	= int(minutes / 60.0)

	return "%02d:%02d:%02d.%03d" % (hours, minutes, seconds, msecs)