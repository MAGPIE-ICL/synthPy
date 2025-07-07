import numpy as np

def open_emi_files(efile):
	with open(efile,'rb') as f:
		line = f.readline()
		# print(line)
		junk = np.fromfile(f,dtype='c',count=9)
		ints = np.fromfile(f,dtype='int16',count=1)
		junk = np.fromfile(f,dtype='c',count=2)
		nTe = np.fromfile(f,dtype='int16',count=1)
		# print("Number of Te points: %i"%nTe)
		junk = np.fromfile(f,dtype='c',count=2)
		nrho = np.fromfile(f,dtype='int16',count=1)
		# print("Number of rho points: %i"%nrho)
		junk = np.fromfile(f,dtype='c',count=2)
		ngrp = np.fromfile(f,dtype='int16',count=1)
		# print("Number of energy groups: %i"%ngrp)
		junk = np.fromfile(f,dtype='c',count=2)
		Te = np.fromfile(f,dtype='float32',count=int(nTe))
		# print(Te)
		#junk = np.fromfile(f,dtype='c',count=2)
		rho = np.fromfile(f,dtype='float32',count=int(nrho))
		# print(rho)
		grps = np.fromfile(f,dtype='float32',count=int(ngrp+1))
		# print(grps)
		emi_data = np.fromfile(f,dtype='float32',count=-1)
		emi_data = emi_data.reshape(int(ngrp),int(nrho),int(nTe))
		# print(emi_data.shape)
	grp_centres = 0.5*(grps[:-1]+grps[1:])
	return grp_centres,grps,rho,Te,emi_data




