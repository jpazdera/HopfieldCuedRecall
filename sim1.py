import numpy as np
from hopfield import HopfieldNetwork
from generate_stimuli import generate_stimuli, pair_with_noise
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_LISTS = 100
    NUM_ITEMS = 100
    NUM_FEATURES = 500

    
    rt = np.zeros(10000)
    list = 0
    while list < NUM_LISTS:
        print "LIST: ", list
        items, pairs = generate_stimuli(NUM_ITEMS, NUM_FEATURES)
        H = HopfieldNetwork(NUM_FEATURES * 2)
        H.train(pairs)
        success, iter = H.cue_recall(pair_with_noise(items[0:1]), pairs)
        if success:
            rt[iter] += 1
        list += 1
    
    rt = rt.reshape(20, 500)
    rt = np.sum(rt, axis=1)
    plt.plot(rt)
    plt.show()