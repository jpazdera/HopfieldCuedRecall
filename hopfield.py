import random
import numpy as np

class HopfieldNetwork():
    def __init__(self, num_neurons):
        self.W = np.zeros((num_neurons, num_neurons), dtype=np.int)
        self.S = np.zeros(num_neurons, dtype=np.int)
    
    def train(self, items):
        for i in range(items.shape[0]):
            self.W += items[i:i+1] * items[i:i+1].T
        np.fill_diagonal(self.W, 0)
    
    def cue_recall(self, item, itemlist=None, iterations=10000):
        self.S[:] = item  # Set the initial state of the network to the probe vector
        half_size = item.size/2
        # node_order = np.random.randint(self.S.size, size=iterations)
        node_order = np.random.randint(half_size, high=self.S.size, size=iterations)  # Randomly select the order in which nodes will be updated
        for i in range(iterations):
            n = node_order[i]
            self.S[n] = np.sign(np.sum(self.W[n] * self.S))  # Update the state of the currently selected node
            
            z = np.where(self.S == 0)[0]  # Find and randomly convert zeros to 1 or -1
            if len(z) > 0:
                self.S[z] = np.array([random.getrandbits(1) for x in range(len(z))]) * 2 - 1
            
            # Check if the state matches any of the words in the lexicon
            for row in range(itemlist.shape[0]):
                #if np.all(self.S[half_size:] == itemlist[row, half_size:]):
                if np.all(self.S == itemlist[row]):
                    return True, i

        return False, i
        