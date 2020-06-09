# use the monte carlo simulation data to do PCA

import math
import json
import numpy as np



############################################################################
##########             FUNCTION  AND CLASS DEFINITION             ##########
############################################################################

def impose_pbc(lattice):
	'''
	impose the periodic boundary condition for a given lattice configuration
	
	Args:
	lattice: a given lattice configuration
	'''
	lattice[0] = lattice[-2]
	lattice[-1] = lattice[1]
	lattice[:, 0] = lattice[:, -2]
	lattice[:, -1] = lattice[:, 1]


def generate_config(length, condition='pbc', direction='random'):
	'''
	generate a spin configuration imposed by obc or pbc
	
	Args:
	length: the linear size of the lattice
	condition: periodic or open boundary condition
	direction: spin configuration of up/down/random type
	'''
	template = []
	if direction == 'random':
		template = [1., -1.]
	elif direction == 'up':
		template = [1.]
	elif direction == 'down':
		template = [-1.]
	else:
		print("Please check your direction type!")

	if condition == 'obc':
		lattice_size = (length, length)
		raw_config = np.random.choice(template, size=lattice_size)
		return raw_config
	elif condition == 'pbc':
		lattice_size = (length+2, length+2)
		raw_config = np.random.choice(template, size=lattice_size)
		impose_pbc(raw_config)
		return raw_config
	else:
		print("Please check your boundary condition!")


def site_energy(row, column, configuration, strength):
	'''
	calculate the local energy for a single site contributed by the interaction with neighbors
	
	Args:
	row, column: position of the spin that you consider
	configuration: the lattice configuration
	strength: value of the interaction strength J
	'''
	s = - strength * configuration[row, column] * (
		configuration[row-1, column] + configuration[row+1, column] + configuration[row, column-1] + configuration[row, column+1])
	return s


class Lattice():
	'''
	A class for 2d square lattice with monte-carlo function
	
	Args:
	length: linear size of lattice
	boundary: 'pbc' or 'obc'
	direction: 'up' or 'down' or 'random'
	temperature: temperature of the system
	strength: coupling strength J

	Funs:
	get_energy_mag(): update the energy and magnetism based on the current configuration
	flip_spin(): do a single step of monte carlo simulation
	'''
	def __init__(self, length, boundary, direction, temperature, strength):
		self.length = length
		self.boundary = boundary
		self.temperature = temperature
		self.strength = strength
		self.configuration = generate_config(length, boundary, direction)
		self.energy = 0
		self.magnetism = 0

	def get_energy_mag(self):
		'''calculate the total energy and magnetism of the self.configuration'''
		s = 0
		m = 0
		for i in range(1, self.length+1):
			for j in range(1, self.length+1):
				s += site_energy(i, j, self.configuration, self.strength)
				m += self.configuration[i, j]
		self.energy = s/2.  # don't forget the 1/2 factor for bond# = 2 * site#
		self.magnetism = m

	def flip_spin(self):
		'''do a single monte carlo simulation'''
		po1 = np.random.randint(self.length) + 1
		po2 = np.random.randint(self.length) + 1
		energy_change = -2 * site_energy(po1, po2, self.configuration, self.strength)

		if np.random.random() < math.exp(-energy_change/self.temperature):  # base on Metropolis Hastings Algorithm
			self.energy += energy_change
			self.magnetism += -2 * self.configuration[po1, po2]
			self.configuration[po1, po2] = -self.configuration[po1, po2]
			if po1 == 1 or po1 == self.length or po2 == 1 or po2 == self.length:
				impose_pbc(self.configuration)
		else:
			pass



#############################################################################
##########                        MAIN PART                        ##########
#############################################################################

# linear size of the system, e.g. N = L**2 is the site number
L = 20

# set the energy unit, e.g. J = 1 for ferromagenetism and J = -1 for anti-ferromagenetism
J = 1

# one MC time: ensure that each spin will be chosen once on average within this 'time'
mct = L**2

# how many MC times to be taken
mc_step = 6000

# number of data we pick and uncorrelated time
num_each_temp = 100
uncorrelate_step = 25  # it should be larger with larger lattice

# lists used to store data points
temperature_data = []
configuration_data = []

# monte carlo for different temperatures
for temperature in np.arange(1.6, 2.91, 0.1):

	# recover the parity symmetry
	for parity in ['up', 'down']:

		ising_lattice = Lattice(L, 'pbc', parity, temperature, J)
		ising_lattice.get_energy_mag()

		for i in range(mc_step):
			for j in range(mct):
				ising_lattice.flip_spin()
			if i % 50 == 0:
				print("\tThe " + str(i) + " Monte Carlo Time For temperature " + str(temperature) + ". PARITY: " + parity)

		print("\tStart to sample at temperature " + str(temperature) + ". PARITY: " + parity)

		# sample after equilibrium
		for i in range(int(num_each_temp/2)):
			for j in range(uncorrelate_step):  # sample uncorrelated points
				for k in range(mct):
					ising_lattice.flip_spin()
			configuration_data.append(list(np.reshape(ising_lattice.configuration[1:-1, 1:-1], L**2)))
			temperature_data.append(temperature)

		print("\tEnd sampling for temperature " + str(temperature) + ". PARITY: " + parity)


print("Get Total " + str(len(configuration_data)) + " configurations!")

save_data = [configuration_data, temperature_data]
with open('monte_carlo_dataset.json', 'w') as mc_file:
	json.dump(save_data, mc_file)
