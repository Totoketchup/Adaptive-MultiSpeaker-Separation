from data.dataset import H5PY_RW
from models import das

if __name__ == "__main__":
	H5 = H5PY_RW()
	H5.open_h5_dataset('test.h5py')
	H5.set_chunk(50)
	H5.shuffle()
	for X, key in H5:
		print X.shape