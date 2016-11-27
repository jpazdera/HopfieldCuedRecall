import numpy as np
from hopfield import HopfieldNetwork
from generate_stimuli import generate_stimuli, pair_with_noise

if __name__ == "__main__":
    
    items = np.array([[1,-1,-1,1,1],[-1,1,-1,-1,1],[1,1,1,-1,-1]])
    H = HopfieldNetwork(5)
    H.train(items)
    
    correct_weights = np.array([[ 0, -1,  1,  1, -1],
                               [-1,  0,  1, -3, -1],
                               [ 1,  1,  0, -1, -3],
                               [ 1, -3, -1,  0,  1],
                               [-1, -1, -3,  1,  0]])
    if np.all(H.W == correct_weights):
        print 'Weights calculated correctly!'
    else:
        print 'Incorrect weights detected: ', np.where(H.W != correct_weights)
    
    for i in range(10):
        print H.cue_recall(np.array([-1,-1,-1,-1,1]), items)
    
        