"""
磁場中の荷電粒子軌道計算プログラム
Charged Particle Trajectory in Magnetic Field

ルンゲ・クッタ法(RK4)を使用してローレンツ力による粒子運動を計算します。
運動方程式: dv/dt = (q/m) * v × B(x)

使用例:
    python particle_trajectory_clean.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable


class MagneticField:
    """磁場クラス - 各種磁場モデルを提供"""

    def __init__(self, field_type: str = "ABC", k: float = 1.0, magnitude: float = 1.0):
        """
        Parameters:
            field_type: 磁場タイプ ('ABC', 'ABC_c', 'mirror')
            k: 波数パラメータ
            magnitude: 磁場強度
        """
        self.field_type = field_type
        self.k = k
        self.magnitude = magnitude
        self.Bc = np.array([0.0, 0.0, 0.0])  # 一様磁場成分

    def get_field(self, x: np.ndarray) -> np.ndarray:
        """位置xでの磁場ベクトルを返す"""
        if self.field_type == "ABC":
            return self._ABC(x)
        elif self.field_type == "ABC_c":
            return self._ABC_with_constant(x)
        elif self.field_type == "mirror":
            return self._mirror(x)
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _ABC(self, x: np.ndarray) -> np.ndarray:
        """ABC磁場 (Arnold-Beltrami-Childress flow)"""
        xx, yy, zz = x[0], x[1], x[2]
        k = self.k
        B = np.array(
            [
                np.cos(k * yy) + np.sin(k * zz),
                np.sin(k * xx) + np.cos(k * zz),
                np.sin(k * yy) + np.cos(k * xx),
            ]
        )
        return self.magnitude * B

    def _ABC_with_constant(self, x: np.ndarray) -> np.ndarray:
        """ABC磁場 + 一様磁場"""
        return self._ABC(x) + self.Bc

    def _mirror(self, x: np.ndarray) -> np.ndarray:
        """ミラー磁場"""
        m = 1.5  # mirror ratio parameter
        r_sq = x[0] ** 2 + x[1] ** 2
        if r_sq < 1e-10:
            r_sq = 1e-10

        B = np.array([-0.5 * m * x[1] / r_sq, -0.5 * m * x[0] / r_sq, m * x[2] + 1.0])
        return self.magnitude * B


class ParticleIntegrator:
    """粒子軌道積分クラス"""

    def __init__(self, magnetic_field: MagneticField, q_over_m: float = 1.0):
        """
        Parameters:
            magnetic_field: 磁場オブジェクト
            q_over_m: 比電荷 q/m
        """
        self.B_field = magnetic_field
        self.q_over_m = q_over_m

    def equation_of_motion(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        運動方程式: dx/dt = v, dv/dt = (q/m) * v × B

        Parameters:
            state: [x, y, z, vx, vy, vz] 状態ベクトル
            t: 時刻

        Returns:
            d(state)/dt
        """
        x = state[0:3]
        v = state[3:6]

        B = self.B_field.get_field(x)
        acceleration = self.q_over_m * np.cross(v, B)

        dstate_dt = np.zeros(6)
        dstate_dt[0:3] = v
        dstate_dt[3:6] = acceleration

        return dstate_dt

    def rk4_step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        4次ルンゲ・クッタ法による1ステップ積分

        Parameters:
            state: 現在の状態ベクトル
            t: 現在時刻
            dt: 時間刻み

        Returns:
            次ステップの状態ベクトル
        """
        k1 = dt * self.equation_of_motion(state, t)
        k2 = dt * self.equation_of_motion(state + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * self.equation_of_motion(state + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * self.equation_of_motion(state + k3, t + dt)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def integrate(
        self, initial_state: np.ndarray, t_max: float, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        軌道を積分

        Parameters:
            initial_state: 初期状態 [x0, y0, z0, vx0, vy0, vz0]
            t_max: 最大積分時間
            dt: 時間刻み

        Returns:
            (times, states): 時刻配列と状態配列
        """
        n_steps = int(t_max / dt) + 1
        times = np.linspace(0, t_max, n_steps)
        states = np.zeros((n_steps, 6))

        states[0] = initial_state

        for i in range(1, n_steps):
            states[i] = self.rk4_step(states[i - 1], times[i - 1], dt)

        return times, states


class ParticleTrajectorySimulation:
    """粒子軌道シミュレーション統合クラス"""

    def __init__(self, field_type: str = "ABC", q_over_m: float = 1.0):
        """
        Parameters:
            field_type: 磁場タイプ
            q_over_m: 比電荷
        """
        self.magnetic_field = MagneticField(field_type=field_type, k=1.0, magnitude=1.0)
        self.integrator = ParticleIntegrator(self.magnetic_field, q_over_m)

    def set_initial_condition(self, x0: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """
        初期条件設定

        Parameters:
            x0: 初期位置 [x, y, z]
            v0: 初期速度 [vx, vy, vz]

        Returns:
            初期状態ベクトル
        """
        return np.concatenate([x0, v0])

    def run(
        self, initial_state: np.ndarray, t_max: float = 10.0, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        シミュレーション実行

        Parameters:
            initial_state: 初期状態
            t_max: 最大時間
            dt: 時間刻み

        Returns:
            (times, states)
        """
        print(f"Starting simulation...")
        print(f"  Field type: {self.magnetic_field.field_type}")
        print(f"  q/m: {self.integrator.q_over_m}")
        print(f"  Time: 0 -> {t_max}, dt = {dt}")

        times, states = self.integrator.integrate(initial_state, t_max, dt)

        # エネルギー保存確認
        energy_initial = 0.5 * np.sum(initial_state[3:6] ** 2)
        energy_final = 0.5 * np.sum(states[-1, 3:6] ** 2)
        energy_error = abs(energy_final - energy_initial) / energy_initial

        print(f"Simulation completed.")
        print(f"  Initial energy: {energy_initial:.6f}")
        print(f"  Final energy: {energy_final:.6f}")
        print(f"  Energy error: {energy_error*100:.4f}%")

        return times, states

    def plot_trajectory(self, states: np.ndarray, save_path: str = None):
        """
        軌道を3Dプロット

        Parameters:
            states: 状態配列
            save_path: 保存パス (Noneなら表示のみ)
        """
        fig = plt.figure(figsize=(12, 5))

        # 3D軌道
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], "b-", linewidth=0.5)
        ax1.scatter(
            states[0, 0],
            states[0, 1],
            states[0, 2],
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax1.scatter(
            states[-1, 0],
            states[-1, 1],
            states[-1, 2],
            c="red",
            s=100,
            marker="x",
            label="End",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Particle Trajectory in Magnetic Field")
        ax1.legend()
        ax1.grid(True)

        # XY平面投影
        ax2 = fig.add_subplot(122)
        ax2.plot(states[:, 0], states[:, 1], "b-", linewidth=0.5)
        ax2.scatter(
            states[0, 0], states[0, 1], c="green", s=100, marker="o", label="Start"
        )
        ax2.scatter(
            states[-1, 0], states[-1, 1], c="red", s=100, marker="x", label="End"
        )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("XY Projection")
        ax2.legend()
        ax2.grid(True)
        ax2.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_energy(self, times: np.ndarray, states: np.ndarray, save_path: str = None):
        """
        エネルギー時間発展プロット

        Parameters:
            times: 時刻配列
            states: 状態配列
            save_path: 保存パス
        """
        # 運動エネルギー計算
        kinetic_energy = 0.5 * np.sum(states[:, 3:6] ** 2, axis=1)
        energy_error = (kinetic_energy - kinetic_energy[0]) / kinetic_energy[0] * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # エネルギー
        ax1.plot(times, kinetic_energy, "b-", linewidth=1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Kinetic Energy")
        ax1.set_title("Energy Conservation Check")
        ax1.grid(True)

        # エネルギー誤差
        ax2.plot(times, energy_error, "r-", linewidth=1)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy Error (%)")
        ax2.set_title("Relative Energy Error")
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Energy plot saved to {save_path}")
        else:
            plt.show()


def example_single_particle():
    """単一粒子の軌道計算例"""
    print("=" * 60)
    print("単一粒子軌道計算例 - ABC磁場")
    print("=" * 60)

    # シミュレーション設定
    sim = ParticleTrajectorySimulation(field_type="ABC", q_over_m=1.0)

    # 初期条件: 位置と速度
    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.1, 0.0, 0.0])
    initial_state = sim.set_initial_condition(x0, v0)

    # シミュレーション実行
    times, states = sim.run(initial_state, t_max=100.0, dt=0.01)

    # 結果プロット
    sim.plot_trajectory(states)
    sim.plot_energy(times, states)


def example_multiple_initial_conditions():
    """複数の初期条件での軌道比較例"""
    print("=" * 60)
    print("複数初期条件での軌道比較例")
    print("=" * 60)

    sim = ParticleTrajectorySimulation(field_type="ABC", q_over_m=1.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 異なる初期位置で計算
    initial_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    colors = ["blue", "red", "green", "orange"]

    for angle, color in zip(initial_angles, colors):
        x0 = np.array([np.cos(angle), np.sin(angle), 0.0])
        v0 = np.array([0.1, 0.0, 0.0])
        initial_state = sim.set_initial_condition(x0, v0)

        times, states = sim.run(initial_state, t_max=50.0, dt=0.01)

        ax.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=color,
            linewidth=0.8,
            label=f"θ={angle*180/np.pi:.0f}°",
        )
        ax.scatter(states[0, 0], states[0, 1], states[0, 2], c=color, s=100, marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Multiple Particle Trajectories")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def example_different_fields():
    """異なる磁場タイプでの比較例"""
    print("=" * 60)
    print("異なる磁場での軌道比較例")
    print("=" * 60)

    field_types = ["ABC", "mirror"]
    fig = plt.figure(figsize=(14, 6))

    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.1, 0.05, 0.0])

    for i, field_type in enumerate(field_types):
        sim = ParticleTrajectorySimulation(field_type=field_type, q_over_m=1.0)
        initial_state = sim.set_initial_condition(x0, v0)

        times, states = sim.run(initial_state, t_max=30.0, dt=0.01)

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(states[:, 0], states[:, 1], states[:, 2], "b-", linewidth=0.5)
        ax.scatter(
            states[0, 0],
            states[0, 1],
            states[0, 2],
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{field_type} Field")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 実行例を選択
    print("\n磁場中の荷電粒子軌道シミュレーション")
    print("Select example:")
    print("  1: 単一粒子軌道")
    print("  2: 複数初期条件での比較")
    print("  3: 異なる磁場での比較")

    example_single_particle()
    example_multiple_initial_conditions()
    example_different_fields()
