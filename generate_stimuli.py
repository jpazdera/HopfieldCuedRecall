import random
import numpy as np

def generate_stimuli(num_items, num_features):
    num_items = num_items - 1 if num_items % 2 != 0 else num_items
    items = np.array([random.getrandbits(1) for i in range(num_items * num_features)], dtype=np.int8).reshape((num_items, num_features)) # Make a random feature vector for each item
    items = 2 * items - 1  # Translate bits into +1, -1 integers
    pairs = items.reshape((num_items / 2, num_features * 2))  # Pair adjacent items into single vectors
    return items, pairs

def generate_noise_vector(num_features):
    return np.array([[random.getrandbits(1) for i in range(num_features)]], dtype=np.int8) * 2 - 1

def pair_with_noise(item):
    return np.concatenate((item, generate_noise_vector(item.shape[1])), axis=1)
    
if __name__ == "__main__":
    items, pairs = generate_stimuli(4, 5)