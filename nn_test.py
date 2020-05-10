from tetris import Tetris
from es import CMAES, SimpleGA, OpenES, PEPG
import numpy as np
import random as rd

MAX_ITERATIONS = 4000
BOARD_WIDTH = 10
ROTATIONS = [0, 90, 180, 270]
RENDER_DELAY = 0.001
N_IN = 4  # Entrees du NN : state = [lines, holes, total_bumpiness, sum_height]
N_OUT = BOARD_WIDTH + 4 # Sorties du nn : x (BOARD_WIDTH), rotation (4)
N_LAYER1 = 7
N_LAYER2 = 7
N_NEURONS = (N_IN*N_LAYER1 + N_LAYER1) + (N_LAYER1*N_LAYER2 + N_LAYER2) + (N_LAYER2*N_OUT + N_OUT)
best_score = 0
FILE_TEST = "best_fitness/best_nn20042613181571.txt"


class FCLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases


class NeuralNetwork:
    # reseau de neurone a 2 hidden layers.
    def __init__(self, n_inputs, n_layer1, n_layer2, n_output):
        self.n_inputs, self.n_layer1, self.n_layer2, self.n_output = n_inputs, n_layer1, n_layer2, n_output
        self.n_neurons = (self.n_inputs*self.n_layer1 + self.n_layer1) + (self.n_layer1 * self.n_layer2 + self.n_layer2) + (self.n_layer2 * self.n_output + self.n_output)

        # self.layer1 = FCLayer(np.zeros((n_layer1, n_inputs)), np.zeros((n_layer1,1)))
        # self.layer2 = FCLayer(np.zeros((n_layer2, n_layer1)), np.zeros((n_layer2,1)))
        # self.output = FCLayer(np.zeros((n_output, n_layer2)), np.zeros((n_output,1)))

        self.layer1 = FCLayer(np.random.random((n_layer1, n_inputs)), np.random.random((n_layer1,1)))
        self.layer2 = FCLayer(np.random.random((n_layer2, n_layer1)), np.random.random((n_layer2,1)))
        self.output = FCLayer(np.random.random((n_output, n_layer2)), np.random.random((n_output,1)))

    def gene_to_nn(self, genes):
        # import du genome dans le nn
        assert (len(genes) == self.n_neurons)
        gene_index = 0
        for x, line in enumerate(self.layer1.weights):
            for y, column in enumerate(line):
                self.layer1.weights[x,y] = genes[gene_index]
                gene_index += 1

        for x, line in enumerate(self.layer1.biases):
            for y, column in enumerate(line):
                self.layer1.biases[x,y] = genes[gene_index]
                gene_index += 1

        for x, line in enumerate(self.layer2.weights):
            for y, column in enumerate(line):
                self.layer2.weights[x,y] = genes[gene_index]
                gene_index += 1

        for x, line in enumerate(self.layer2.biases):
            for y, column in enumerate(line):
                self.layer1.biases[x,y] = genes[gene_index]
                gene_index += 1

        for x, line in enumerate(self.output.weights):
            for y, column in enumerate(line):
                self.output.weights[x,y] = genes[gene_index]
                gene_index += 1

        for x, line in enumerate(self.output.biases):
            for y, column in enumerate(line):
                self.output.biases[x,y] = genes[gene_index]
                gene_index += 1


        assert (gene_index == self.n_neurons)

    def compute(self, inputs):
        inputs = np.asarray(inputs)
        inputs = np.reshape(inputs, (self.n_inputs, 1))
        x = np.dot(self.layer1.weights, inputs) + self.layer1.biases
        x = np.dot(self.layer2.weights, x) + self.layer2.biases
        x = np.dot(self.output.weights, x) + self.output.biases
        return x

def getSecond(t):
    return t[1]

def choose_action(next_actions, result_nn):
    choice_list = []
    for x_index, x_prob in enumerate(result_nn[:BOARD_WIDTH]):
        for r_index, r_prob in enumerate(result_nn[BOARD_WIDTH:]):
            # print(x_prob)
            choice_list.append(((x_index, ROTATIONS[r_index]), (x_prob * r_prob)[0]))

    choice_list.sort(key = getSecond, reverse = True)

    for action, prob in choice_list:
        if(action in next_actions):
            return action




def play_tetris(ann, render):
    # Entrees du NN : state = [lines, holes, total_bumpiness, sum_height]
    # Sorties du nn : x (BOARD_WIDTH), rotation (4)
    env = Tetris()
    done = False
    obs = env.reset()

    while not(done):
        next_states = env.get_next_states()
        possible_action = [item[0] for item in next_states.items()]
        result_nn = ann.compute(obs)
        chosen_action = choose_action(possible_action, result_nn)

        best_action = None
        for action, state in next_states.items():
            if action == chosen_action:
                best_action = action
                obs = state
                break

        reward, done, final_score = env.play(best_action[0], best_action[1], render=render, render_delay=RENDER_DELAY)


    return final_score


def evaluate(solution, render):
    global best_score
    ann = NeuralNetwork(N_IN, N_LAYER1, N_LAYER2, N_OUT)
    ann.gene_to_nn(solution)
    final_score = play_tetris(ann, render)
    print("BEST_SCORE = ", final_score)
    return final_score






def main():
    init = np.loadtxt(FILE_TEST)
    evaluate(init, True)

if __name__ == '__main__':
    main()
