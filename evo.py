import numpy as np
from numpy.core.defchararray import rindex
from tqdm.std import tqdm

class Evo():
    
    def __init__(self, k_short, num_paths, num_pop, pos, mut_rate=0.1, fitness_alg='lifetime'):
        self.k_short = k_short
        self.num_genes = len(k_short)
        self.num_paths = num_paths
        self.num_pop = num_pop
        self.mut_rate = mut_rate

        self.fitness_alg = fitness_alg  # Algorithm to calculate the fitness
        self.min_algs = ['energy']

        self.pos = pos  # Nodes positions
    
    def create_pop(self):
        self.population = np.random.randint(0, self.num_paths, size=(self.num_pop, self.num_genes))
        self.cost = np.zeros((self.num_genes,self.num_paths,self.num_genes))

        for i in range (0,self.num_genes):
            for j in range (0,self.num_paths):
                for k in range (1,len(self.k_short[i,j])):
                    if k == len(self.k_short[i,j])-1:
                        self.cost[i,j,self.k_short[i,j][k] - 1] = 10
                    else:
                        self.cost[i,j,self.k_short[i,j][k] - 1] = 20
                    travel = 0.04*(np.linalg.norm(np.array(self.pos[self.k_short[i,j][k]]) - np.array(self.pos[self.k_short[i,j][k-1]])))**2
                    self.cost[i,j,self.k_short[i,j][k] - 1] += travel
    

    def fitness(self):
        self.energy = np.zeros((self.num_pop,self.num_genes))
        for i in range(0,self.num_pop):
            for j in range(0,self.num_genes):
                self.energy[i] += self.cost[j,self.population[i,j]]
        
        # Choose between different fitness algorithms
        if self.fitness_alg == 'lifetime':
            # Lifetime based
            self.energy = 10000 / np.max(self.energy, axis=1)
        elif self.fitness_alg == 'energy':
            # Energy based
            self.energy[np.arange(len(self.energy)), np.argmax(self.energy, axis=1)] **= 2 # Mágica - Fez funcionar
            self.energy = self.energy**2
            self.energy = np.sum(self.energy, axis=1)
        else:
            print('Fitness algorithm not recognized. Using default (Lifetime based)')
            self.energy = 10000 / np.max(self.energy, axis=1)


    def lifetime(self, individual): 
        energy_individual = np.zeros((self.num_genes))
        for i in range(0,self.num_genes):
            energy_individual += self.cost[i,individual[i]]
        lifetime = 10000/(np.max(energy_individual))
        return lifetime, np.argmax(energy_individual)+1, energy_individual

    def reproduction(self):
        # Invert energy if needed to be a maximization problem
        if self.fitness_alg in self.min_algs:
            inv_energy = (1 / self.energy)
            repro_chance = inv_energy / np.sum(inv_energy)
        else:
            repro_chance = self.energy / np.sum(self.energy)

        repro_chance = np.rint(repro_chance * self.num_pop * 10).astype(int)        
        prob_array = np.repeat(self.population, repro_chance, axis=0)

        # TODO: Keep the 20% best individuals
        new_pop = [self.population[np.argmax(repro_chance)]]
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
        t_otm = tqdm(range(num_iter))
        for i in t_otm:
            self.fitness()
            best, fitness = self.reproduction()
            graf.append(fitness)
            self.mutate()
            lifetime, node, all_energies = self.lifetime(best)
            t_otm.set_postfix(fitness='{:.2f}'.format(fitness), lifetime='{:.2f}'.format(lifetime), node=str(node))
            # print('Best in iteration {}: '.format(i), best, ' Fitness: ', fitness, 'Total lifetime: ',lifetime,'for node: ',node)

        return best, graf, lifetime, np.sum(all_energies), node

