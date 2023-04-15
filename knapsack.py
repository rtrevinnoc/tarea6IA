import itertools, functools
from typing import Sequence
import time
import matplotlib.pyplot as plt
import random

# Clase para construir nuestros experimentos
class Experiment:
    def __init__(self, problem, populations, generations):
        self.steps = populations
        self.trials = generations
        self.problem = problem

        self.currentStep = 1
        self.increment = 1

        self.solutionsPerStep = []
        self.avgTimesPerStep = []
    
    def runStepGenetic(self, step):
        print("\n# STEP " + str(self.currentStep))
        times = []
        solutions = []
        for x in self.trials:
            start = time.time()

            ex = Knapsack(self.problem)
            bestKnapsack = ex.solveGenetic(step, x);
            self.optimum = ex.optimum
            currentSolution = ex.knapsackProfit(bestKnapsack) 
            print(ex.calculateAccuracy(bestKnapsack), ex.knapsackWeight(bestKnapsack), currentSolution, self.optimum, bestKnapsack)
            solutions.append(currentSolution)

            end = time.time()
            currentTime = (end - start) * 1000
            times.append(currentTime)
            print("\n# TRIAL " + str(x) + " took " + str(currentTime) + " s")
        avgTime = sum(times) / len(times)
        self.solutionsPerStep.append(solutions)
        self.avgTimesPerStep.append(avgTime)

        print("\n\n# STEP TIMES:", times, "AVG: " + str(avgTime) + " s")

    def run(self, increment=1):
        self.increment = increment
        for step in self.steps:
            # self.runStepAStar(step)
            # self.runStepDFS(step)
            self.runStepGenetic(step)
            self.currentStep += 1

    def plot(self):
        fig, ax = plt.subplots(1, len(self.steps), sharey=True, sharex=True)

        for index, step in enumerate(self.steps):
            ax[index].bar([str(trial) for trial in self.trials], self.solutionsPerStep[index])
            ax[index].title.set_text(step)
            ax[index].set_ylim(self.optimum * 0.65, self.optimum)

        fig.text(0.5, 0.04, 'Tamaño de Población', ha='center', va='center')
        fig.text(0.06, 0.5, 'Numero de Generaciones', ha='center', va='center', rotation='vertical')
        fig.suptitle(f'Resultados de Ejecucion de algoritmo genetico (optimo: {self.optimum})')
        plt.show()

class Knapsack:
    def __init__(self, problemPath: str) -> None:
        with open(f'./problems/{problemPath}') as file:
            parsedLines = [dict(zip(('profit', 'weight'), map(float, line.strip().split(" ")))) for line in file.readlines()]
            head = parsedLines.pop(0)

            self.n = head['profit']
            self.wmax = head['weight']
            self.items = parsedLines

        folder, file = problemPath.split("/")
        optimumPath = folder + "-optimum/" + file 
        with open(f'./problems/{optimumPath}') as file:
            self.optimum = float(file.readline().strip())

        # print(self.n, self.wmax, self.items, [x['weight'] for x in self.items])

    def knapsackWeight(self, knapsack: Sequence) -> float:
        if len(knapsack) > 1:
            return functools.reduce(lambda knapsackWeight, item: knapsackWeight + item['weight'], knapsack, 0)
        elif len(knapsack) == 1:
            return knapsack[0]['weight']
        else:
            return 0

    def knapsackProfit(self, knapsack: Sequence) -> float:
        if len(knapsack) > 1:
            return functools.reduce(lambda knapsackProfit, item: knapsackProfit + item['profit'], knapsack, 0)
        elif len(knapsack) == 1:
            return knapsack[0]['profit']
        else:
            return 0

    def clipKnapsack(self, knapsack: Sequence) -> list:
        clippedKnapsack = []
        clippedKnapsackWeight = 0
        for item in knapsack:
            newWeight = clippedKnapsackWeight + item['weight']
            if newWeight <= self.wmax:
                clippedKnapsackWeight = newWeight
                clippedKnapsack.append(item)
        return clippedKnapsack

    def verifySolution(self, knapsack) -> bool:
        return self.knapsackProfit(knapsack) == self.optimum

    def calculateAccuracy(self, knapsack) -> float:
        return (self.knapsackProfit(knapsack) / self.optimum) * 100

    def solveGenetic(self, pop_size, num_generations):
        # Crear poblacion aleatoria
        population = []
        for _ in range(pop_size):
            chromosome = random.sample(self.items, random.randint(0, len(self.items) - 1))
            population.append(chromosome)

        # Verificar la adaptabilidad de la poblacion
        for generation in range(num_generations):
            fitness_scores = []
            for chromosome in population:
                total_value = self.knapsackProfit(chromosome)#sum([item['profit'] for i, item in enumerate(self.items) if chromosome[i]])
                total_weight = self.knapsackWeight(chromosome)#sum([item['weight'] for i, item in enumerate(self.items) if chromosome[i]])
                if total_weight > self.wmax:
                    fitness_scores.append(0)
                else:
                    fitness_scores.append(total_value)

            # Crar parejas
            parents = []
            for i in range(pop_size // 2):
                parent1 = random.choices(population, weights=fitness_scores)[0]
                parent2 = random.choices(population, weights=fitness_scores)[0]
                parents.append((parent1, parent2))

            # Combinar parejas y crear hijos con mutaciones
            offspring = []
            for parent1, parent2 in parents:
                combined = parent1 + [x for x in parent2 if x not in parent1]
                # if random.random() < 0.5:
                    # child = random.sample(combined, random.randint(0, len(parent1) - 1))#random.shuffle(combined)
                # else:
                child = random.sample(combined, random.randint(0, len(combined) - 1))#random.shuffle(combined)
                offspring.append(child)

            # Reemplazar menos adaptados con hijos
            combined_population = list(zip(population, fitness_scores))
            combined_population.sort(key=lambda x: x[1], reverse=True)
            population = [x[0] for x in combined_population]

        # Calcular el mejor adaptado
        best_fitness = 0
        best_solution = []
        for chromosome in population:
            total_value = self.knapsackProfit(chromosome)#sum([item['profit'] for i, item in enumerate(self.items) if chromosome[i]])
            total_weight = self.knapsackWeight(chromosome)#sum([item['weight'] for i, item in enumerate(self.items) if chromosome[i]])
            if total_weight <= self.wmax and total_value > best_fitness:
                best_fitness = total_value
                best_solution = chromosome
        return best_solution

problems = ["low-dimensional/f4_l-d_kp_4_11", "low-dimensional/f3_l-d_kp_4_20", "low-dimensional/f6_l-d_kp_10_60", "low-dimensional/f7_l-d_kp_7_50", "low-dimensional/f9_l-d_kp_5_80", "low-dimensional/f8_l-d_kp_23_10000"]
for problem in problems:
    ex = Experiment(problem, [20, 40, 60, 80], [20, 40, 60, 80, 100])
    ex.run()
    ex.plot()
