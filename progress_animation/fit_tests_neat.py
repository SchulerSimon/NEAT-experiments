import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


n_points = 200


def gen_rand_point(x_range: tuple, y_range: tuple) -> tuple:
    return (random.uniform(*x_range), random.uniform(*y_range))


def classifying_function(x: float, y: float) -> bool:
    return y > 1 / 2 * math.sin(4 * x)

points = [gen_rand_point((-0.9, 0.9), (-0.9, 0.9)) for _ in range(n_points)]
pos_p = [p for p in points if classifying_function(*p)]
neg_p = [p for p in points if not classifying_function(*p)]

# start neat
import neat

inputs = pos_p + neg_p
outputs = [(1.0,)] * len(pos_p) + [(0.0,)] * len(neg_p)

# prepare animation
nets = []
last_fitness = -100
fig = plt.figure(figsize=(10, 10))  # instantiate a figure to draw
axes = plt.axes()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= ((output[0] - xo[0]) ** 2) / n_points
        
    fitnesses = [(genome.fitness, genome) for genome_id, genome in genomes]
    fitnesses.sort(key=lambda x : x[0], reverse=True)
    fitness = fitnesses[0][0]
    global last_fitness
    if fitness > last_fitness:
        last_fitness = fitness
        genome = fitnesses[0][1]
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

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
try:
    # Run until a solution is found.
    winner = p.run(eval_genomes)
    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))
    print("\nOutput:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
except KeyboardInterrupt:
    print("aborted")

end_offset = 10 #how many still images to show with the winning net
num_frames = len(nets)

def animate(i):
    print(f"frame {i+1}/{num_frames + end_offset}")
    if i >= num_frames:
        i = num_frames - 1
    X = np.arange(-1, 1, 0.01)
    Y = np.arange(-1, 1, 0.01)
    Z = np.empty((X.shape[0], Y.shape[0]))

    for n, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[j, n] = float(nets[i].activate((x, y))[0])

    pos_p = [p for p in points if classifying_function(*p)]
    neg_p = [p for p in points if not classifying_function(*p)]
    x1, y1 = zip(*pos_p)
    x2, y2 = zip(*neg_p)
    
    axes.clear()
    axes.set_xticks([])
    axes.set_xticklabels([])
    
    axes.set_title(f"improvement number {i+1}")
    axes.contour(X, Y, Z)
    axes.scatter(x1, y1, marker=".", color="green")
    img = axes.scatter(x2, y2, marker=".", color="red")
    return [img]

# do the animation
anim = animation.FuncAnimation(fig, animate, frames=num_frames + end_offset, interval=750, blit=True)
anim.save('neat.gif',writer='imagemagick')