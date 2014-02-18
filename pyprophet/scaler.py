#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except:
    profile = lambda x: x


class Scaler:
	
	def scale(self, in_data):
		raise NotImplementedError


class NonScaler(Scaler):
	
	def scale(self, in_data):
		return in_data


class ShiftDivScaler(Scaler):
	
	def __init__(self, shift, div):
		self.shift = shift
		self.div = div
	
	def scale(self, in_data):
		return (in_data - self.shift) / self.div
