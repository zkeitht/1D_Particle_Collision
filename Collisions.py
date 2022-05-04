from email.base64mime import header_length
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import itertools
from scipy import stats

def night():
    t = time.localtime().tm_hour
    if t<7 or t>20:
        return True
    else:
        return False
if night():
    plt.style.use('dark_background')

gap_factor = 1.2 # trial and error - not bad to prevent "sticking" when used with particle radius of 0.5

class Box():
    def __init__(self, dim, size = 20, dt = 0.01):
        self.dim  = dim
        self.size = size
        self.dt   = dt
        self.t    = 0
        self.particles = []
        self.tot_kinetic  = []
        self.tot_momentum = []
        self.fig, self.ax = plt.subplots()

    def after_draw(self, i, calc_tot = False):
        """Draws the movement of the particles after simulating and storing all the frames"""
        self.ax.clear()
        # plot particles and wall
        for particle in self.particles:
            s = particle.radius
            if type(particle).__name__ == "Wall":
                s = 0.5
            self.ax.plot(particle.positions[i], 
                        0, 
                        color = particle.color, 
                        marker = particle.shape,
                        markersize = s * 13)
        # shows energy and momentum
        if calc_tot:
            self.ax.text(self.size*0.6, 0.04, 
                        f"Total KE: {self.tot_kinetic[i]:.1f}\nTotal mo: {self.tot_momentum[i]:.1f}")
        self.ax.set_xlim(-self.size, self.size)
        # self.ax.set_ylim(0, self.t + 10)
    
    def overall_draw(self, ax = None, start = 0):
        """Plots overall (static) paths of all particles against time after the simulation"""
        if ax == None:
            ax = self.ax
        for count, particle in enumerate(self.particles):
            linewidth = 1.5
            if count == 5:
                linewidth= 5
            ax.plot(
                    # np.arange(0, self.t + self.dt - self.dt, self.dt), # for Q5
                    np.linspace(0, self.t, len(particle.positions)),
                    particle.positions,
                    color = particle.color,
                    linewidth = linewidth)
        ax.set_ylim(-self.size-3, self.size+3)
    
    def update(self, calc_tot = False):
        for particle in self.particles:
            particle.move()
        # if calc_tot:
        #     self.calc_tot()
        self.t += 1
    
    def collisions(self):
        combs = list(itertools.combinations(range(len(self.particles)), 2))
        for pair in combs:
            self.particles[pair[0]].check_n_collide(self.particles[pair[1]])

    def particles_only(self):
        l = [p for p in self.particles if type(p).__name__ == "Particle"]
        return l

    def calc_tot(self):
        tot_kinetic  = 0
        tot_momentum = 0
        for p in self.particles_only():
            tot_kinetic  += p.calc_kinetic()
            tot_momentum += p.calc_momentum()
        self.tot_kinetic.append(tot_kinetic)
        self.tot_momentum.append(tot_momentum)

class Particle():
    def __init__(self, Box, position = 0, velocity = 1, inv_mass = 0.5,  radius = 0.5, 
                color = 'r', shape = "o"):
        self.position  = position
        self.inv_mass  = inv_mass
        self.velocity  = velocity
        if self.inv_mass != 0: # not wall: calculate momentum and kinetic energy
            self.momentum  = self.position/self.inv_mass
            self.kinetic   = (self.velocity**2)/2/self.inv_mass
        self.radius    = radius
        self.color     = color
        self.shape     = shape
        self.positions  = []
        self.velocities = []

        self.Box      = Box
        self.Box.particles.append(self)

        ## to prevent speed that is too high - prevent sticking
        # if abs(self.velocity) * self.Box.dt >= self.radius:
        #     print("Box speed limit achieved, velocity reduced")
        #     self.velocity = self.radius/self.Box.dt * 0.9

    def move(self):
        self.position = np.add(self.position, self.velocity*self.Box.dt)
        self.positions.append(self.position)
        self.velocities.append(self.velocity)
        
    def check_n_collide(self, other):
        if np.linalg.norm(np.subtract(self.position, other.position)) <= (self.radius + other.radius)*gap_factor:
            # m1u1, m2u2, -> m1v1, m2v2
            # v1 = [(m1-m2)u1 + 2 m2u2]/(m1+m2) = [(i2-i1)u1 + 2 i1u2]/(i1 + i2)
            # v2 = [2 m1u1 + (m2-m1)u2]/(m1+m2) = [2 i2u1 + (i1-i2)u2]/(i1 + i2)
            u1, u2 = self.velocity, other.velocity
            red_mass = self.inv_mass + other.inv_mass
            if red_mass != 0: # not two walls "colliding"
                self.velocity  = ((other.inv_mass - self.inv_mass)*u1 + 2 * self .inv_mass * u2)/red_mass
                other.velocity = ((self.inv_mass - other.inv_mass)*u2 + 2 * other.inv_mass * u1)/red_mass
    
    def calc_kinetic(self):
        return (self.velocity**2)/self.inv_mass/2
    
    def calc_momentum(self):
        return self.velocity/self.inv_mass

class Wall(Particle):
    def __init__(self, Box, position, velocity = 0, inv_mass = 0, radius = 0.8, 
                color = 'tab:gray', shape = "|"):
        super().__init__(Box, position, velocity, inv_mass, radius, color, shape)
        self.momentum = None


def init_BoxPtcWall(N, velocities, masses):
    """Initialises N particles in a Box, with two walls"""
    Box1 = Box(dim = 1)
    ptcs = []
    walls = []
    
    p_range = int(Box1.size*0.9)
    pos = [i for i in range(-p_range+2, p_range-1, int(Box1.size*1.5/N))]
    colors = ["pink", "r", "tab:red", "orange", "y", "c", "tab:green", "g", "tab:blue", "b", "tab:purple"]
    for i, x, v, m, color in zip(range(N), pos, velocities, masses, colors):
        ptc = Particle(Box1, x, v, 1/m, color = color)
        ptcs.append(ptc)

    for i, x in zip(range(2), [-p_range, p_range]):
        walls.append(Wall(Box1, x))
    return (Box1, ptcs, walls)

###_______________________________________________________
### 3. Thermal equilibrium: relative velocity for given mass ratio
print("\nThermal equilibrium: relative velocity for given mass ratio")
three = input("(press Enter to view, any other key to skip: )")
if three == "":
# if True:
    # initialising 10 unequal masses with order across 10 magnitudes, random velocities
    N = 10
    masses     = [1, 4, 10, 40, 50, 100, 400, 1000, 4000, 5000]
    random.shuffle(masses)
    velocities = [random.random()-0.5 for i in range(N)]
    Box1, ptcs, walls = init_BoxPtcWall(N, velocities, masses)
    f = 50000 # a long period
    # f = 100 # fast run to check stuff
    print("Running simulation...")
    for i in range(f): # run simulation for f frames
            Box1.update(calc_tot = False)
            Box1.collisions()

    playspeed = int(0.25/Box1.dt)

    if True:
        def animate(i):
            Box1.after_draw(i*playspeed, calc_tot = False)
        anim = animation.FuncAnimation(Box1.fig, animate, frames = int(f/playspeed), interval = 1)
        plt.show()
    
    if True: ### plot position histograms
        fig, axs = plt.subplots(3, 4)
        i = 0
        for row in range(3):
            for col in range(4):
                p = ptcs[i]
                axs[row, col].hist(p.positions, color = p.color)
                axs[row, col].set_title(f"M = {1/p.inv_mass}")
                axs[row, col].set_xlim(-Box1.size, Box1.size)
                i += 1
                if i>=len(ptcs):
                    break
        axs[2,2].axis("off")
        axs[2,3].axis("off")
        fig.tight_layout()
        txt = """
        Particles with really heavy neighbours 
        have relatively constricted positions.
        """
        plt.figtext(0.5, 0.12, txt, wrap=True, horizontalalignment='left')
        fig.suptitle("Positions")
        plt.show()  

    if True: ### plot velocity histograms
        fig, axs = plt.subplots(3, 4)
        ptcs.sort(key = lambda x: x.inv_mass, reverse = True)
        i = 0
        for row in range(3):
            for col in range(4):
                p = ptcs[i]
                axs[row, col].hist(p.velocities, color = p.color)
                axs[row, col].set_title(f"M = {1/p.inv_mass}")
                # axs.set_title(f"M = {1/p.inv_mass}")
                i += 1
                if i>=len(ptcs):
                    break
        axs[2,2].axis("off")
        axs[2,3].axis("off")
        fig.suptitle("Velocities")
        txt = """
        If thermal equilibrium is achieved, the average kinetic energy of all particles 
        should be the same, hence a pair of particles whose masses are in the ratio 4:1 
        should have a (maximum) velocity ratio of 1:2 as observed. 
        (see velocity spread of masses: 1 and 4, 10 and 40, 100 and 400, 1000 and 4000)
        """
        plt.figtext(0.5, 0.06, txt, wrap=True, horizontalalignment='left')
        fig.tight_layout()
        plt.show()

###_______________________________________________________
### 4. Adiabatic compression: 1D adiabatic index (gamma)
print("\nAdiabatic compression: 1D adiabatic index (gamma)")
four = input("(press Enter to view, any other key to skip: )")
if four == "":
# if True:
    # initialising 10 equal masses, random velocities
    N = 10
    masses     = [1]*N
    velocities = [random.random()-0.5 for i in range(N)]
    Box1, ptcs, walls = init_BoxPtcWall(N, velocities, masses)
    # move piston towards right
    walls[0].velocity = 1
    f = 3000
    print("Running simulation...")
    for i in range(f): # run simulation for f frames
            Box1.update(calc_tot = False)
            Box1.collisions()
    playspeed = int(0.25/Box1.dt)
    if True: # animation
        def animate(i):
            Box1.after_draw(i*playspeed, calc_tot = False)
        anim = animation.FuncAnimation(Box1.fig, animate, frames = int(f/playspeed), interval = 1)
        plt.show()
    

    ## pV^gamma = const:
    # p = N*m*mean(v^2) prop to mean(v^2)/x; V in 1D prop to x = wall2.position - wall1.position 
    all__v_sqr = [np.array(ptc.velocities)**2 for ptc in ptcs]
    dist = np.array(walls[1].positions) - np.array(walls[0].positions)
    pressure_prop = np.mean(all__v_sqr, axis = 0) / dist

    y = np.log(pressure_prop, out=np.zeros_like(pressure_prop), where=(pressure_prop!=0))
    x = np.log(dist, out=np.zeros_like(y), where=(y!=0))
    fig, ax = plt.subplots()
    ax.scatter(x, y, s = 5, label = "ln p against ln V")

    #regression part
    frac = 0.15 # take only the final fraction of the p and v values for regression calculation
    xp, yp = x[int(len(x)*(1-frac)):], y[int(len(x)*(1-frac)):]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xp,yp)
    # plotting best fit for final compression
    line = slope*xp+intercept
    ax.plot(xp, line, 'r', label='y={:.3f}x+{:.3f}'.format(slope,intercept))

    # arrow of compression
    arrx, arry = 2.4, -0.7
    ax.arrow(arrx, arry, -0.3, 0, width = 0.1, head_length = 0.05)
    txt = "Direction \nof compression"
    plt.figtext(0.3, 0.4, txt)

    txt = """
    Should expect a 1D adiabatic index to be 3, 
    as Cv =R/2 and Cp = 3R/2 for 1 degree of freedom.
    Since pV = gamma = 3,
    plot of ln p against ln V gives a gradient of ~-3.
    """
    plt.figtext(0.5, 0.1, txt, wrap=True, horizontalalignment='center')
    ax.set_ylabel("ln p")    
    ax.set_xlabel("ln V")

    plt.legend()
    plt.show()

###_______________________________________________________
### 5. Piston Equilibrium
print("\nPiston equilibrium: piston separating particles on both sides")
five = input("(press Enter to view, any other key to skip: )")
if five == "":
# if True:
    # initialising masses (1 heavy in centre) and velocities (random except for centre)
    masses     = [1.1, 1.2, 1.1, 1.3, 1.1, 100.0, 10.1, 10.2, 10.1, 10.8, 10.4]
    velocities = [random.random()*-0.5 for i in range(11)]
    Box1, ptcs, walls = init_BoxPtcWall(11, velocities, masses)
    ptcs[5].velocity = 2
    f = 50000
    # f = 5000
    print("Running simulation...")
    for i in range(f): # run simulation for f frames
            Box1.update(calc_tot = False)
            Box1.collisions()

    playspeed = int(0.25/Box1.dt)

    if True:
        def animate(i):
            Box1.after_draw(i*playspeed, calc_tot = False)
        anim = animation.FuncAnimation(Box1.fig, animate, frames = int(f/playspeed), interval = 1)
        plt.show()
    if True:
        fig, ax = plt.subplots()
        Box1.overall_draw(ax = ax)
        plt.title(f"Oscillating piston separating two ideal gases.")
        txt = """
        Given enough time, pressure and temperature on both sides should equilibrate, 
        and the volume occupied by each side should be roughly equal,
        as V=nRT/p and n, T, p on both sides are equal.
        The piston is observed to oscillate sinusoidally due to random fluctuations.
        """
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center')
        fig.subplots_adjust(bottom=0.2) 
        plt.show()


# ### plotting Kinetic energy as a funtion of time
# # plt.plot(np.arange(0, Box1.t-Box1.dt, Box1.dt), Box1.tot_kinetic)
# # plt.show()
# def mean_chunks(l, n):
#     return [np.mean(chunk) for chunk in np.array_split(l,n)]

# n = 100
# """
# E_chunks_mean = []
# for ptc in ptcs:
#     E = np.array([v**2 for v in ptc.velocities])/ptc.inv_mass/2
#     # np.append(E_chunks, np.array(np.array_split(E, n)))
#     # E_chunks.append(np.array_split(E, n))
#     E_chunks_mean.append(mean_chunks(E, n))
    
# print(len(E_chunks_mean[0]))
# print(E_chunks_mean[0])
# for count, ptc in enumerate(ptcs):
#     plt.plot(
#             mean_chunks(np.linspace(0, Box1.t, len(ptc0.velocities)), n),
#             E_chunks_mean[count],
#             )
# plt.title("Individual energies")
# plt.show()

# stdE = np.std(E_chunks_mean, axis = 0) # deviation of all particle energies as time progresses
# plt.plot(np.linspace(0, Box1.dt, len(stdE)), stdE)
# plt.title("Energies deviation as a function of time")
# plt.show()"""