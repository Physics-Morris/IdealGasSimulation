from typing import Collection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from tqdm import tqdm
import scipy.stats as stats

kb = np.float64(1.3806488e-23)  # j/k (+/- 1.3e-29)
spatial_res = 1000

#> Maxwell-Boltzmann distribution
def Maxwell_Boltzmann(mass, temperature):
    vx = np.sqrt(kb * temperature / mass) * \
         np.random.normal(np.float64(0), np.float64(1))
    vy = np.sqrt(kb * temperature / mass) * \
         np.random.normal(np.float64(0), np.float64(1))
    return np.array([vx, vy], dtype=np.float64)

class particle:
    
    def __init__(self, x, y, vx, vy, mass, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.size = size

class box:

    def __init__(self, L, W):
        self.L = L
        self.W = W
        self.particles = []
        self.step = 0
        self.time = 0.0
        self.x = []
        self.y = []
        self.vx = []
        self.vy = []
        self.Momentum = []
        self.Energy = []
        self.Temperature = []
        self.param = []
        

    def Random_Position(self, size):
        while True:
            x = np.random.rand() * (self.L - size) + size / 2
            y = np.random.rand() * (self.W - size) + size / 2
            #> check wether overlap
            flag = False
            for particle in self.particles:
                if ((particle.x - x)**2 + (particle.y - y)**2)**0.5 <= size + particle.size:
                    flag = True
                    break
            if flag == False: break
        return np.array([x, y])

    def add_particles(self, number, mass, size, distribution, E_tot):
        if (size**2 * number > 0.5 * self.W * self.L):
            print('Too much particle added (not enough space). Consider increase volumn \
                   or decrease particle size')
        # print('Allocating particles')
        if distribution == 'Maxwell-Boltzmann':
            # v_mean = 2 * np.sqrt(kb * Temp / mass) * np.sqrt(2 / np.pi)
            v_mean = np.sqrt(E_tot / np.float(number) * 2 / mass)
            Temp = (v_mean / np.sqrt(2 / np.pi) / 2)**2 * mass / kb
            for i in tqdm(range(number)):
                [vx, vy] = Maxwell_Boltzmann(mass, Temp)
                [x, y] = self.Random_Position(size)
                self.particles.append(particle(x, y, vx, vy, mass, size))
        elif distribution == 'Uniform':
            E_mean = E_tot / np.float(number)
            for i in tqdm(range(number)):
                theta = np.random.rand() * 2 * np.pi
                E_rand = np.random.rand() * 2 * E_mean
                v = np.sqrt(E_rand * 2 / mass)
                vx = v * np.cos(theta)
                vy = v * np.sin(theta)
                [x, y] = self.Random_Position(size)
                self.particles.append(particle(x, y, vx, vy, mass, size))
        elif distribution == 'One-Non-Zero':
            v = np.sqrt(E_tot * 2 / mass)
            theta = np.random.rand() * 2 * np.pi
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            [x, y] = self.Random_Position(size)
            self.particles.append(particle(x, y, vx, vy, mass, size))
            for i in tqdm(range(number)):
                [x, y] = self.Random_Position(size)
                self.particles.append(particle(x, y, 0, 0, mass, size))
        else:
            print("Unreconginize option for distribution")

    def get_data(self):
        x, y, vx, vy, v = [], [], [], [], []
        for i in range(len(self.particles)):
            x.append(self.particles[i].x)
            y.append(self.particles[i].y)
            vx.append(self.particles[i].vx)
            vy.append(self.particles[i].vy)
            v.append(np.sqrt(self.particles[i].vx**2+self.particles[i].vy**2))
        return x, y, vx, vy, v

    def collide(self):
        #> preform collition
        for i in range(len(self.particles)):
            xi = self.particles[i].x
            yi = self.particles[i].y
            ri = self.particles[i].size / 2.0
            mi = self.particles[i].mass
            for j in range(i+1, len(self.particles)):
                xj = self.particles[j].x
                yj = self.particles[j].y
                rj = self.particles[j].size / 2.0
                mj = self.particles[i].mass
                #> distance between two particles
                d = ((xi - xj)**2 + (yi - yj)**2)**0.5
                if d  <= (ri + rj):
                    vi = np.array([self.particles[i].vx, self.particles[i].vy])
                    vj = np.array([self.particles[j].vx, self.particles[j].vy])
                    rij = (np.array([xi, yi]) - np.array([xj, yj])) / d
                    q = (-2.0 * mi * mj / (mi + mj)) * np.dot((vi - vj), rij) * rij
                    [self.particles[i].vx, self.particles[i].vy] = vi + q / mi
                    [self.particles[j].vx, self.particles[j].vy] = vj - q / mj
                    break

    def move(self, dt):
        #> move particles
        for i in range(len(self.particles)):
            self.particles[i].x += self.particles[i].vx * dt
            self.particles[i].y += self.particles[i].vy * dt
        
        self.time += dt

    def wall(self):
        delta_P = 0
        #> check wall
        for i in range(len(self.particles)):
            x = self.particles[i].x
            y = self.particles[i].y
            size = self.particles[i].size
            if (x + size / 2 >= self.L or x - size / 2 <= 0): 
                self.particles[i].vx *= -1
            if (y + size / 2 >= self.W or y - size / 2 <= 0): 
                self.particles[i].vy *= -1
            #> measure pressure
            if x + size / 2 >= self.L:
                delta_P += 2 * self.particles[i].mass * abs(self.particles[i].vx)
        return delta_P


    def measure(self):
        P_tot = 0
        E_tot = 0
        for particle in self.particles:
            P_tot += particle.mass * np.sqrt(particle.vx**2 + particle.vy**2)
            E_tot += 0.5 * particle.mass * (particle.vx**2 + particle.vy**2)

        #> fit temperature
        vx, vy = np.array(self.vx[int(0)]), np.array(self.vy[int(0)])
        v = np.sqrt(vx**2 + vy**2)
        maxwell = stats.maxwell
        params = maxwell.fit(v, floc=0)
        kbT = params[1]**2 * self.particles[0].mass
        T = E_tot / len(self.particles) / kb
        return P_tot, E_tot, T, params


    def store_data(self):
        x, y, vx, vy, v = self.get_data()
        self.x.append(x)
        self.y.append(y)
        self.vx.append(vx)
        self.vy.append(vy)

    def start(self, step):
        #> find optimal dt
        v = []
        for i in range(len(self.particles)):
            v.append(np.sqrt(self.particles[i].vx**2 + self.particles[i].vy**2))
        dt = max(self.L, self.W) / spatial_res / max(v)

        delta_P = 0
        delta_T = 0
        for i in tqdm(range(step)):
            self.store_data()
            self.collide()
            self.move(dt)
            dp = self.wall()
            Mom, E, Temperature, params = self.measure()
            self.Momentum.append(Mom)
            self.Temperature.append(Temperature)
            self.Energy.append(E)
            self.param.append(params)
            self.step += 1
            delta_T += dt
            delta_P += dp
        Pressure = delta_P / delta_T / self.W
        return self.Temperature[0], Pressure

    def plot_simple(self, step):
        import matplotlib as mpl
        from pylab import cm
        def get_size(self):
            s = []
            for i in range(len(self.particles)):
                s.append(self.particles[i].size)
            return np.array(s)
        mpl.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['legend.fontsize'] = 13
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.linewidth'] = 1
        x, y = self.x[int(step-1)], self.y[int(step-1)]
        vx, vy = np.array(self.vx[int(step-1)]), np.array(self.vy[int(step-1)])
        v = np.sqrt(vx**2 + vy**2)
        fig = plt.figure(figsize=(13.5/1.5, 4.5/1.5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        dist = ax1.hist(v, density=True)
        ax1.set_xlabel('$v$')
        ax1.set_ylabel('$f(v)$')
        title = 'velocity distribution'
        ax1.set_title(title)

        #> fit temperature
        xx = np.linspace(0, max(v)*1.5, 250)
        ax1.plot(xx, stats.maxwell.pdf(xx, *self.param[step-1]), lw=1)

        sc = ax2.scatter(x, y, c=v, vmin=min(v), vmax=max(v), s=4*get_size(self))
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        ax2.set_aspect('equal')
        title = 'particles distribution'
        ax2.set_title(title)
        plt.colorbar(sc)

        plt.show()

    def animation(self, interval, step):
        fig = plt.figure(figsize=(13.5/1.5, 4.5/1.5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        vx, vy = np.array(self.vx[int(0)]), np.array(self.vy[int(0)])
        v = np.sqrt(vx**2 + vy**2)
        hst = ax1.hist(v)
        ax1.set_xlabel('$v$')
        ax1.set_ylabel('$f(v)$')

        xdata, ydata = [], []
        ax2.set_xlim(0, self.L)
        ax2.set_ylim(0, self.W)
        scat = ax2.scatter(xdata, ydata, vmin=0, vmax=1)
        cbar = plt.colorbar(scat)
        ax2.set_aspect('equal')
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(r'$v/v_{max}$', rotation=90)
        

        def get_size(self):
            s = []
            for i in range(len(self.particles)):
                s.append(self.particles[i].size)
            return np.array(s)

        def update(frame):
            xdata, ydata = self.x[int(frame)], self.y[int(frame)]
            vx, vy = np.array(self.vx[int(frame)]), np.array(self.vy[int(frame)])
            v = np.sqrt(vx**2 + vy**2)
            scat.set_offsets(np.array([xdata, ydata]).transpose())
            scat.set_sizes(4**get_size(self))
            scat.set_array(v / max(v))
            ax1.clear()
            ax1.hist(v, density=True)
            #> fit temperature
            xx = np.linspace(0, max(v)*1.5, 250)
            ax1.plot(xx, stats.maxwell.pdf(xx, *self.param[frame]), lw=1)
            return scat

        ani = FuncAnimation(fig=fig, func=update, frames=np.arange(0, self.step-1, step), interval=interval)
        plt.show()  
        return ani
    
    def save_animation(self, animation, name):
        f = name + ".gif" 
        writergif = PillowWriter(fps=30) 
        animation.save(f, writer=writergif)