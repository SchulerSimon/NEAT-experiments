import random
import math
import numpy as np
import matplotlib.pyplot as plt

n_points = 100


def gen_rand_point(x_range: tuple, y_range: tuple) -> tuple:
    return (random.uniform(*x_range), random.uniform(*y_range))

def classifying_function_0(x: float, y: float) -> bool:
    # return x ** 2 + y ** 2 >= 0.5
    return x > y

def classifying_function_1(x: float, y: float) -> bool:
    # return x ** 2 + y ** 2 >= 0.5
    return x+y > 0

def classifying_function_2(x: float, y: float) -> bool:
    # return x ** 2 + y ** 2 >= 0.5
    return y > 1 / 2 * math.sin(4 * x)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 21))
for row, classifying_function in enumerate([classifying_function_0, classifying_function_1, classifying_function_2]):
    points = [gen_rand_point((-0.9, 0.9), (-0.9, 0.9)) for _ in range(n_points)]
    X = np.arange(-1, 1, 0.01)
    Y = np.arange(-1, 1, 0.01)
    Z = np.empty((X.shape[0], Y.shape[0]))

    for n, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[j, n] = float(classifying_function(x, y))

    pos_p = [p for p in points if classifying_function(*p)]
    neg_p = [p for p in points if not classifying_function(*p)]
    x1, y1 = zip(*pos_p)
    x2, y2 = zip(*neg_p)


    
    axes[row][0].contour(X, Y, Z)
    axes[row][0].scatter(x1, y1, marker=".", color="green")
    axes[row][0].scatter(x2, y2, marker=".", color="red")
    axes[row][0].set_title("optimal classification")

    # start neat
    import neat

    inputs = pos_p + neg_p
    outputs = [(1.0,)] * len(pos_p) + [(0.0,)] * len(neg_p)


    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(inputs, outputs):
                output = net.activate(xi)
                genome.fitness -= ((output[0] - xo[0]) ** 2) / n_points


    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.ini",
    )
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    # Run until a solution is found.
    winner = p.run(eval_genomes)
    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))
    print("\nOutput:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    for n, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[j, n] = float(winner_net.activate((x, y))[0])

    pos_p = [p for p in points if classifying_function(*p)]
    neg_p = [p for p in points if not classifying_function(*p)]
    x1, y1 = zip(*pos_p)
    x2, y2 = zip(*neg_p)

    axes[row][1].contour(X, Y, Z)
    axes[row][1].scatter(x1, y1, marker=".", color="green")
    axes[row][1].scatter(x2, y2, marker=".", color="red")
    axes[row][1].set_title("NEAT classification")

plt.savefig('result.png')
plt.show()