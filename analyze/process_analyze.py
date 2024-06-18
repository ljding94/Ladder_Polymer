import numpy as np
import scipy.fft

# some data process


def transform_pair_distribution_data_to_structure_factor(filename):
    data = np.genfromtxt(filename, skip_header=5, delimiter=",", filling_values=np.nan)
    print("data",data)
    gr = data[0, 1:]
    Sq = scipy.fft.dct(gr)
    q = np.array(range(len(Sq))*np.pi/len(Sq))
    print("Sq",Sq)
    content = np.genfromtxt(filename, delimiter=",", filling_values=np.nan)

    content[5, 0], content[5, 1:] = "q", q
    content[6, 0], content[6, 1:] = "Sq", Sq
    np.savetxt(filename[:-4]+"Sq", content, delimiter=",")
    return Sq
