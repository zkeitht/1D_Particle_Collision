import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import itertools
from scipy import stats

# from pytest import param

def night():
    t = time.localtime().tm_hour
    if t<7 or t>21:
        return True
    else:
        return False
if night():
    plt.style.use('dark_background')

class Box():
    def __init__(self, dim, size = 10):
        self.dim  = dim
        self.size = size
        self.t    = 0
        self.dt   = 0.005

        self.particles = []
        self.tot_kinetic  = []
        self.tot_momentum = []
        self.fig, self.ax = plt.subplots()

    def after_draw(self, i, calc_tot = True):
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
                        markersize = s * 35)
        # shows energy and momentum
        if calc_tot:
            self.ax.text(self.size*0.6, 0.04, 
                        f"Total KE: {self.tot_kinetic[i]:.1f}\nTotal mo: {self.tot_momentum[i]:.1f}")
        self.ax.set_xlim(-self.size, self.size)
        # self.ax.set_ylim(0, self.t + 10)
    
    def overall_draw(self, start = 0):
        for count, particle in enumerate(self.particles):
            linewidth = 1.5
            if count == 5:
                linewidth= 5
            self.ax.plot(
                        # np.arange(0, self.t + self.dt - self.dt, self.dt), # for Q5
                        np.linspace(0, self.t, len(particle.positions)),
                        particle.positions,
                        color = particle.color,
                        linewidth = linewidth)
        self.ax.set_ylim(-self.size, self.size)
    
    def update(self, calc_tot = True):
        for particle in self.particles:
            particle.move()
        if calc_tot:
            self.calc_tot()
        self.t += self.dt
    
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
    def __init__(self, 
                Box, 
                position = 0, 
                velocity = 1,
                inv_mass = 0.5, 
                radius = 0.2, 
                color = 'r',
                shape = "o"):
        self.position  = position
        self.inv_mass  = inv_mass
        self.velocity  = velocity
        if self.inv_mass != 0:
            self.momentum  = self.position/self.inv_mass
            self.kinetic   = (self.velocity**2)/2/self.inv_mass
        self.radius    = radius
        self.color     = color
        self.shape     = shape
        self.positions  = [self.position]
        self.velocities = [self.velocity]

        self.Box      = Box
        self.Box.particles.append(self)

        if abs(self.velocity) * self.Box.dt >= self.radius:
            print("Box speed limit achieved, velocity reduced")
            self.velocity = self.radius/self.Box.dt * 0.9

    def move(self):
        self.position = np.add(self.position, self.velocity*self.Box.dt)
        self.positions.append(self.position)
        self.velocities.append(self.velocity)
        
    def check_n_collide(self, other):
        if np.linalg.norm(np.subtract(self.position, other.position)) <= (self.radius + other.radius):
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
    def __init__(self, 
                Box, 
                position,
                velocity = 0, 
                inv_mass = 0,
                radius = 0.01, 
                color = 'tab:gray',
                shape = "|"):
        super().__init__(Box, position, velocity, inv_mass, radius, color, shape)
        self.momentum = None


def plot_position_hist(ptcs):
    ### plotting position histograms
    fig, axs = plt.subplots(3, 4)
    # ptcs.sort(key = lambda x: x.inv_mass, reverse = True)
    i = 0
    for row in range(3):
        for col in range(4):
            p = ptcs[i]
            axs[row, col].hist(p.positions, color = p.color)
            axs[row, col].set_title(f"M = {1/p.inv_mass}")
            axs[row, col].set_xlim(-p_range, p_range)
            # axs.set_title(f"M = {1/p.inv_mass}")
            i += 1
            if i>=len(ptcs):
                break
    fig.tight_layout()
    fig.suptitle("Positions")
    plt.show()

def plot_velocity_hist(ptcs):
    ### plotting velocity histograms
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
    fig.tight_layout()
    fig.suptitle("Velocities")
    plt.show()

def Q5_Pressure(ptcs, wall1, wall2):
    ### pV^gamma = const:
    # p = N*m*mean(v^2) prop to mean(v^2)/x; V in 1D prop to x = wall2.position - wall1.position 
    all__v_sqr = [np.array(ptc.velocities[1:])**2 for ptc in ptcs]
    dist = np.array(wall2.positions[1:]) - np.array(wall1.positions[1:])
    pressure_prop = np.mean(all__v_sqr, axis = 0) / dist

    x = np.log(pressure_prop, out=np.zeros_like(pressure_prop), where=(pressure_prop!=0))
    y = np.log(dist, out=np.zeros_like(x), where=(x!=0))

    plt.scatter(
            x,
            y,
            s = 5,
            label = "ln(p)-ln(V)"   
            )

    #regression part
    xp, yp = x[int(len(x)*0.2):], y[int(len(x)*0.2):]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xp,yp)

    line = slope*xp+intercept
    plt.plot(xp, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    #end

    plt.legend()
    # plt.plot(, (v**2)*(x**3))
    plt.show()



p_range = 10

Box1 = Box(dim = 1, size=10)
rand_pos = [i for i in range(-p_range+2, p_range-1)]
# random.shuffle(rand_pos)
ptc0 = Particle(Box1,
                color = "tab:pink"
                )
ptc1 = Particle(Box1,
                color = "r"
                )
ptc2 = Particle(Box1,
                color = "tab:red"
                )
ptc3 = Particle(Box1,
                position = rand_pos[3],
                color = "tab:orange"
                )
ptc4 = Particle(Box1,
                color = "y"
                )
ptc5 = Particle(Box1,
                color = "c"
                )
ptc6 = Particle(Box1,
                color = "tab:green"
                )
ptc7 = Particle(Box1,
                color = "g"
                )
ptc8 = Particle(Box1,
                color = "tab:blue"
                )
ptc9 = Particle(Box1,
                color = "b"
                )
ptc10 = Particle(Box1,
                color = "tab:purple"
                )

# ptc11 = Particle(Box1,
#                 )
# ptc12 = Particle(Box1,
#                 )
# ptc13 = Particle(Box1,
#                 )
# ptc14 = Particle(Box1,
#                 )
# ptc15 = Particle(Box1,
#                 )
# ptc16 = Particle(Box1,
#                 )
# ptc17 = Particle(Box1,
#                 )
# ptc18 = Particle(Box1,
#                 )

wall1 = Wall(Box1,
            position = -p_range,
            # velocity = 1.5 # - compression
            )
wall2 = Wall(Box1,
            position = p_range,
            # velocity = 0
            )

ptcs = [eval("ptc"+str(i)) for i in range(11)]
# masses = [1, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 7, 8]
# masses = [2000, 5, 10, 20, 50, 100, 200, 500, 1000, 5000] # 3.1: 4 orders of magnitude
# masses = [5, 600, 250, 500, 700, 300, 400, 800, 450, 550, 5] # 3.2: light ends
# masses = [350, 200, 150, 250, 300, 100, 5, 10, 7, 8, 4] # 3.3: left heavy right light
# masses = [1]*len(ptcs) # 4: ideal gas compression - same mass
masses = [1.1, 1.2, 1.1, 1.3, 1.1, 100.0, 10.1, 10.2, 10.1, 10.8, 10.4]
# masses = [1.1, 1.2, 1.1, 1.3, 1.1, 100.0, 4.1, 4.2, 4.1, 4.8, 4.4]
# masses = [1.1, 1.2, 1.1, 1.3, 1.1, 100.0, 2.1, 2.2, 2.1, 2.8, 2.4]
# masses = [5*m for m in masses]
imasses = [1/m for m in masses]
# idx = 5
# imasses.append(imasses[idx]/4)
a, b = 0, 0
# a, b = imasses[idx], imasses[-1]
# random.shuffle(imasses)

# setting initial velocities, positions, and masses
iv = 1.5
for ptc, x, im, v in zip(ptcs, range(-p_range+3, p_range-1, 1), imasses, range(len(ptcs))):
    ptc.position = x
    ptc.inv_mass = im
    if random.randint(0,1) == 1:
        ptc.velocity = -iv
    else:
        ptc.velocity = iv

ptc5initv = 1 # 5: oscillating piston
ptc5.velocity = ptc5initv # 5: oscillating piston
ptc1to4 = [p for p in ptcs if p.inv_mass in [a, b]]

# run simulation for f * dt seconds
Box1.dt = 0.005
f = 30000
for i in range(f):
        Box1.update(calc_tot = False)
        Box1.collisions()

animate = True
animate = False
playspeed = 15

if animate:
    def animate(i):
        Box1.after_draw(i*playspeed, calc_tot = False)
    anim = animation.FuncAnimation(Box1.fig, animate, frames = int(f/playspeed), interval = 1)
    plt.show()
else: # No animation
    Box1.overall_draw()
    plt.title(f"f={f}, dt={Box1.dt}, piston_iv={ptc5initv}, other_iv={iv}")
    plt.show()

# for p in [ptc0, ptc1, ptc2, ptc]
for p in ptc1to4:
    plt.hist(p.velocities, color = p.color)
    plt.title(f"M = {1/p.inv_mass}")
    plt.show()

### plotting Kinetic energy as a funtion of time
# plt.plot(np.arange(0, Box1.t-Box1.dt, Box1.dt), Box1.tot_kinetic)
# plt.show()
def mean_chunks(l, n):
    return [np.mean(chunk) for chunk in np.array_split(l,n)]

n = 100
"""
E_chunks_mean = []
for ptc in ptcs:
    E = np.array([v**2 for v in ptc.velocities])/ptc.inv_mass/2
    # np.append(E_chunks, np.array(np.array_split(E, n)))
    # E_chunks.append(np.array_split(E, n))
    E_chunks_mean.append(mean_chunks(E, n))
    
print(len(E_chunks_mean[0]))
print(E_chunks_mean[0])
for count, ptc in enumerate(ptcs):
    plt.plot(
            mean_chunks(np.linspace(0, Box1.t, len(ptc0.velocities)), n),
            E_chunks_mean[count],
            )
plt.title("Individual energies")
plt.show()

stdE = np.std(E_chunks_mean, axis = 0) # deviation of all particle energies as time progresses
plt.plot(np.linspace(0, Box1.dt, len(stdE)), stdE)
plt.title("Energies deviation as a function of time")
plt.show()"""



### Answers:
# plot_position_hist(ptcs) # plotting position histograms
# plot_velocity_hist(ptcs) # plotting velocity histograms
# Q5_Pressure(ptcs, wall1, wall2)
