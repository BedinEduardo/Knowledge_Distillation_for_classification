# code to load a customized logger
import time

class Bcolors(object):
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

class timeLogger(object)      :
	def __init__(self):
		self.record_t = None
		self.bcolors = Bcolors()
		print() #misc
	
	def print(self, msg = None):
		if msg is not None:
			if self.record_t is None:
				self.record_t = time.time()
			print(self.bcolors.HEADER + msg + self.bcolors.ENDC, end='')
		
		else:
			if self.record_t is not None:
				cost = round(time.time() - self.record_t, 4)
				print("... [{}success{}][{}{}seconds]".format(self.bcolors.OKCYAN, self.bcolors.ENDC, self.bcolors.OKGREEN, cost, self.bcolors.ENDC))
			
			else:
				print("....[{}skipped{}]".format(self.bcolors.WARNING, self.bcolors.ENDC))
			# cost = round(time.time() - self.record_t, 4)
			# print("....[{}success{}][{}{}seconds]".format(self.bcolors.OKCYAN, self.bcolors.ENDC, self.bcolors.OKGREEN, cost, self.bcolors.ENDC))
		self.record_t = None