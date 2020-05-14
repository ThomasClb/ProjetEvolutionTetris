from tetris import Tetris
from es import CMAES, SimpleGA, OpenES, PEPG
import numpy as np
import random as rd
import sys, os

MAX_ITERATIONS = 4000
BOARD_WIDTH = 10
ROTATIONS = [0, 90, 180, 270]
RENDER_DELAY = 0.001
N_IN = 4  # Entrees du NN : state = [lines, holes, total_bumpiness, sum_height]
N_OUT = 1 # Sorties du nn : score_du_next_state
N_LAYER1 = 24 # nombre de neurones dans la couche 1
N_LAYER2 = 24
N_NEURONS = (N_IN*N_LAYER1 + N_LAYER1) + (N_LAYER1*N_LAYER2 + N_LAYER2) + (N_LAYER2*N_OUT + N_OUT) # nbre total de neurones
best_score = -1000
best_fitness = -1000
FILE_TEST = "best_score_score/best_nn89857.txt"


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
        # onvertit liste de genes en neural network
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


        assert (gene_index == self.n_neurons) # Pour s'assurer qu'on a bien extrait tout le genome

    def compute(self, inputs):
        inputs = np.asarray(inputs)

        inputs = np.reshape(inputs, (self.n_inputs, 1)) # transformation de liste en vecteur numpy
        x = np.dot(self.layer1.weights, inputs) + self.layer1.biases
        x = np.dot(self.layer2.weights, x) + self.layer2.biases
        x = np.dot(self.output.weights, x) + self.output.biases
        return x


def getSecond(t):
    return t[1]


def choose_action(next_actions, result_nn):
    # Reseau a 14 sorties
    choice_list = []
    for x_index, x_prob in enumerate(result_nn[:BOARD_WIDTH]):
        for r_index, r_prob in enumerate(result_nn[BOARD_WIDTH:]):
            choice_list.append(((x_index, ROTATIONS[r_index]), (x_prob * r_prob)[0]))

    # On trie la liste pour avoir les meilleures combinaisons (x,rotation) en premier
    choice_list.sort(key = getSecond, reverse = True)

    # On choisit la meilleure combinaison qui est autorisée (en pratique, la première action dans choice_list qui est aussi dans next_action)
    for action, prob in choice_list:
        if(action in next_actions):
            return action



def choose_action(ann, next_states):
    best_score_next_state = -10000
    best_action_next_state = None

    for action, state in next_states.items():
        score = ann.compute(state) # Le NN attribue un score au next_state
        if(score > best_score_next_state):
            best_score_next_state = score
            best_action_next_state = action

    return best_action_next_state




def play_tetris(ann, render):
    # Entrees du NN : state = [lines, holes, total_bumpiness, sum_height]
    # Sorties du nn : x (BOARD_WIDTH), rotation (4)
    env = Tetris()
    done = False
    obs = env.reset()

    while not(done):
        next_states = env.get_next_states()
        chosen_action = choose_action(ann, next_states)

        best_action = chosen_action
        # for action, state in next_states.items():
        #     if action == chosen_action:
        #         best_action = action
        #         break

        reward, done, final_score = env.play(best_action[0], best_action[1], render=render, render_delay=RENDER_DELAY)
        obs = next_states[best_action]

    return final_score - obs[1], obs[1]


def evaluate(solution, render, save = True):
    global best_score
    ann = NeuralNetwork(N_IN, N_LAYER1, N_LAYER2, N_OUT)
    ann.gene_to_nn(solution) # Conversion du gene en NN
    final_score, holes = play_tetris(ann, render)

    # Pour garder le NN du meilleur score
    if(final_score > best_score):
        print("[SAVE SCORE] BEST_SCORE = ", final_score, "HOLES : ", holes)
        best_score = final_score
        if(save):
            np.savetxt("best_score_score/best_nn" + str(best_score) + ".txt", solution)

    return final_score





def test_solver(solver):
    # Execution du solver (CMAES, OpenAI...)
    global best_fitness
    history = []

    # Code dans simple_es_example.ipynb
    for j in range(MAX_ITERATIONS):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)

        for i in range(solver.popsize):
            # Juste pour parfois voir comment joue le programme
            if(i == 0 and j % 10 == 0):
                render = True
            else:
                render = False
            fitness_list[i] = evaluate(solutions[i], render)

        solver.tell(fitness_list)
        result = solver.result()
        history.append(result[1])

        print(j, "Fitness : ", result[1], "Mean : ", np.mean(fitness_list), "Std : ", np.mean(result[3]))

        # Pour conserver le NN du meilleur fitness
        if(j % 20 == 0 or result[1] > best_fitness):
            print("[SAVE FITNESS] BEST_FITNESS = ", result[1])
            best_fitness = result[1]
            np.savetxt("best_fitness_score/best_nn" + str(int(result[1])) + "_{}.txt".format(j), result[0])







def main():
    commands = sys.argv

    if("solve" in commands):
        solver = CMAES(num_params = N_NEURONS,
                    sigma_init=0.50,       # initial standard deviation
                    popsize=1000,           # population size
                    weight_decay=0.01)     # weight decay coefficient

        # solver = SimpleGA(N_NEURONS)
        test_solver(solver)
    elif("test" in commands):
        init = np.loadtxt(FILE_TEST)
        evaluate(init, True, False)
        # input()

if __name__ == '__main__':
    main()
