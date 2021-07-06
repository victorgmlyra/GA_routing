from networkx.classes.function import selfloop_edges
import numpy as np
from numpy.core.defchararray import rindex

class Evo():
    # Energy based
    def __init__(self, k_short, num_paths, num_pop, pos, mut_rate=0.1):
        self.k_short = k_short
        self.num_genes = len(k_short)
        self.num_paths = num_paths
        self.num_pop = num_pop
        self.mut_rate = mut_rate

        self.pos = pos  # Nodes positions
    
    def create_pop(self):
        self.population = np.random.randint(0, self.num_paths, size=(self.num_pop, self.num_genes))
        self.cost = np.zeros((self.num_genes,self.num_paths,self.num_genes))

        # print(self.population)
        # print('-----------------------------------')

        for i in range (0,self.num_genes):
            for j in range (0,self.num_paths):
                for k in range (1,len(self.k_short[i,j])):
                    if k == len(self.k_short[i,j])-1:
                        self.cost[i,j,self.k_short[i,j][k] - 1] = 10
                    else:
                        self.cost[i,j,self.k_short[i,j][k] - 1] = 20
                    travel = 0.04*(np.linalg.norm(np.array(self.pos[self.k_short[i,j][k]]) - np.array(self.pos[self.k_short[i,j][k-1]])))**2
                    self.cost[i,j,self.k_short[i,j][k] - 1] += travel
                    # print(travel, 'nodes', self.k_short[i,j][k], 'and', self.k_short[i,j][k-1])
    

    def fitness(self):
        self.energy = np.zeros((self.num_pop,self.num_genes))
        for i in range(0,self.num_pop):
            for j in range(0,self.num_genes):
                self.energy[i] += self.cost[j,self.population[i,j]]
        
        self.energy = self.energy**2
        self.energy = np.sum(self.energy, axis=1)


    def reproduction(self):
        inv_energy = (1 / self.energy)
        repro_chance = inv_energy / np.sum(inv_energy)
        repro_chance = np.rint(repro_chance * self.num_pop * 10).astype(int)
        # print(repro_chance)
        
        prob_array = np.repeat(self.population, repro_chance, axis=0)

        new_pop = [self.population[np.argmax(repro_chance)]]
        # new_pop = []
        while(len(new_pop) != self.num_pop):
            r_int = np.random.randint(0, len(prob_array), 2)
            p1 = prob_array[r_int[0]]
            p2 = prob_array[r_int[1]]
            if not (p1 == p2).all():    # Diferent
                cross_point = np.random.randint(1, len(p1)-1)
                new_ind = np.zeros_like(p1)
                new_ind[:cross_point] = p1[:cross_point]
                new_ind[cross_point:] = p2[cross_point:]
                new_pop.append(new_ind)
        
        self.population = np.array(new_pop)
        return (self.population[0], self.energy[np.argmax(repro_chance)])
        
    def mutate(self):
        random_pop = np.random.randint(0, self.num_paths, size=(self.num_pop, self.num_genes))
        mut_chances = np.random.rand(self.num_pop, self.num_genes)
        mut_chances[0] = np.ones((1, self.num_genes))
        self.population = np.where(mut_chances > self.mut_rate, self.population, random_pop)        

    def fit(self, num_iter):
        self.create_pop()
        graf = []
        for i in range(num_iter):
            self.fitness()
            best, fitness = self.reproduction()
            graf.append(fitness)
            self.mutate()
            print('Best in iteration {}: '.format(i), best, ' Fitness: ', fitness)
        # print('Final Population:')
        # print(self.population)
        return best,graf

if __name__ == '__main__':
    kk = np.array([
        [[0, 2, 3], [0, 2, 3]],
        [[0, 3, 4], [0, 4, 1]],
        [[0, 3, 4], [0, 4, 1]],
        [[0, 3, 4], [0, 4, 1]],
        [[0, 3, 4], [0, 4, 1]],
    ])

    mama = Evo(kk,2,3)
    mama.create_pop()
    # mama.reproduction()
    mama.mutate()