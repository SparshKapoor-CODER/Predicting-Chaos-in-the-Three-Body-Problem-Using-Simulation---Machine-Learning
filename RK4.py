import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

G = 1  # Gravitational constant

class ThreeBodySimulator3D:
    def __init__(self, masses, positions, velocities, dt=0.005):
        self.masses = np.array(masses)
        self.n = 3
        self.dt = dt
        self.pos = np.array(positions, dtype=float)  # shape (3, 3)
        self.vel = np.array(velocities, dtype=float)
        self.trajectories = [self.pos.copy()]
        self.energies = [self.compute_energy()]

    def compute_accelerations(self, pos):
        acc = np.zeros_like(pos)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    r = pos[j] - pos[i]
                    acc[i] += G * self.masses[j] * r / np.linalg.norm(r) ** 3
        return acc

    def rk4_step(self):
        dt = self.dt

        def derivatives(p, v):
            a = self.compute_accelerations(p)
            return v, a

        p1, v1 = self.pos, self.vel
        dp1, dv1 = derivatives(p1, v1)

        p2, v2 = p1 + 0.5 * dt * dp1, v1 + 0.5 * dt * dv1
        dp2, dv2 = derivatives(p2, v2)

        p3, v3 = p1 + 0.5 * dt * dp2, v1 + 0.5 * dt * dv2
        dp3, dv3 = derivatives(p3, v3)

        p4, v4 = p1 + dt * dp3, v1 + dt * dv3
        dp4, dv4 = derivatives(p4, v4)

        self.pos += (dt / 6) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        self.vel += (dt / 6) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        self.trajectories.append(self.pos.copy())
        self.energies.append(self.compute_energy())

    def run(self, steps=100000):
        for _ in range(steps):
            self.rk4_step()

    def get_trajectories(self):
        return np.array(self.trajectories)
    
    def compute_energy(self):
        # Kinetic Energy
        ke = 0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1))

        # Potential Energy
        pe = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                r = np.linalg.norm(self.pos[i] - self.pos[j])
                pe -= G * self.masses[i] * self.masses[j] / r

        return ke + pe

# -------- Run & Animate --------
if __name__ == "__main__":
    masses = [1.0, 0.1, 0.1]  # Central body is heavy, others are light
    positions = [[0, 0, 0],[1, 0, 0],[0, 1, 0]]
    velocities = [[0, 0, 0],[0, 1, 0],[-1, 0, 0]]

    sim = ThreeBodySimulator3D(masses, positions, velocities)
    sim.run(steps=3000)
    traj = sim.get_trajectories()

    # Animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title("3D Three-Body Simulation (Animated)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Create points and trails
    points = [ax.plot([], [], [], 'o', label=f"Body {i+1}")[0] for i in range(3)]
    trails = [ax.plot([], [], [], '-', lw=1)[0] for _ in range(3)]
    ax.legend()

    def update(frame):
        for i in range(3):
            # Current position
            x, y, z = traj[frame, i]
            points[i].set_data([x], [y])
            points[i].set_3d_properties([z])
            # Trail
            trails[i].set_data(traj[:frame, i, 0], traj[:frame, i, 1])
            trails[i].set_3d_properties(traj[:frame, i, 2])
        return points + trails

    ani = FuncAnimation(fig, update, frames=len(traj), interval=10, blit=True)
    
    # Energy plot
    plt.figure()
    plt.plot(sim.energies)
    plt.title("Total Energy Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Energy")
    plt.grid(True)
    
    plt.show()





# Infinity sign

masses = [1.0, 1.0, 1.0]
positions = [[-0.970, 0.243, 0], [0.0, 0.0, 0], [0.970, -0.243, 0]]
velocities = [[0.466, 0.432, 0], [-0.932, -0.865, 0], [0.466, 0.432, 0]]




# Central body is heavy others are light
masses = [1.0, 0.001, 0.001]  # Central body is heavy, others are light
positions = [[0, 0, 0],[1, 0, 0],[0, 1, 0]]
velocities = [[0, 0, 0],[0, 1, 0],[-1, 0, 0]]

