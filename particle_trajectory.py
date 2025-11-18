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


class PhysicalConstants:
    """物理定数クラス"""

    # 基本定数
    c = 2.998e8  # 光速 [m/s]
    e = 1.602e-19  # 電気素量 [C]
    m_e = 9.109e-31  # 電子質量 [kg]
    m_p = 1.673e-27  # 陽子質量 [kg]

    # 電子の比電荷 [C/kg]
    q_over_m_electron = e / m_e  # ≈ 1.759e11 C/kg
    # 陽子の比電荷 [C/kg]
    q_over_m_proton = e / m_p  # ≈ 9.578e7 C/kg

    @staticmethod
    def cyclotron_frequency(B: float, q_over_m: float) -> float:
        """サイクロトロン周波数 ω_c = qB/m [rad/s]"""
        return q_over_m * B

    @staticmethod
    def larmor_radius(v_perp: float, B: float, q_over_m: float) -> float:
        """ラーモア半径 r_L = v⊥/(ω_c) = m*v⊥/(qB) [m]"""
        omega_c = PhysicalConstants.cyclotron_frequency(B, q_over_m)
        return v_perp / omega_c if omega_c != 0 else np.inf


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

    def __init__(
        self,
        magnetic_field: MagneticField,
        q_over_m: float = 1.0,
        particle_name: str = "particle",
    ):
        """
        Parameters:
            magnetic_field: 磁場オブジェクト
            q_over_m: 比電荷 q/m [C/kg]
            particle_name: 粒子名 (表示用)
        """
        self.B_field = magnetic_field
        self.q_over_m = q_over_m
        self.particle_name = particle_name

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

    def __init__(
        self,
        field_type: str = "ABC",
        q_over_m: float = 1.0,
        particle_name: str = "particle",
    ):
        """
        Parameters:
            field_type: 磁場タイプ
            q_over_m: 比電荷 [C/kg]
            particle_name: 粒子名
        """
        self.magnetic_field = MagneticField(field_type=field_type, k=1.0, magnitude=1.0)
        self.integrator = ParticleIntegrator(
            self.magnetic_field, q_over_m, particle_name
        )
        self.particle_name = particle_name

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
        print(f"  Particle: {self.particle_name}")
        print(f"  Field type: {self.magnetic_field.field_type}")
        print(f"  q/m: {self.integrator.q_over_m:.3e} C/kg")
        print(f"  B magnitude: {self.magnetic_field.magnitude:.3e} T")

        # サイクロトロン周波数を計算
        omega_c = PhysicalConstants.cyclotron_frequency(
            self.magnetic_field.magnitude, self.integrator.q_over_m
        )
        print(f"  Cyclotron freq: {omega_c:.3e} rad/s ({omega_c/(2*np.pi):.3e} Hz)")
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
        fig = plt.figure(figsize=(14, 6))

        # 3D軌道
        ax1 = fig.add_subplot(121, projection="3d")
        # 時間経過を色で表現
        time_colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        for i in range(len(states) - 1):
            ax1.plot(
                states[i : i + 2, 0],
                states[i : i + 2, 1],
                states[i : i + 2, 2],
                color=time_colors[i],
                linewidth=0.5,
                alpha=0.6,
            )
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
        ax1.set_title(f"Particle Trajectory ({len(states)} points)")
        ax1.legend()
        ax1.grid(True)
        # カラーバー追加
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(states))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label("Time Step", rotation=270, labelpad=15)

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


def example_electron():
    """電子の軌道計算例 (実際の物理パラメータ)"""
    print("=" * 60)
    print("電子軌道計算例 - ABC磁場 (実物理パラメータ)")
    print("=" * 60)

    # 電子の比電荷を使用
    q_over_m = PhysicalConstants.q_over_m_electron  # 1.759e11 C/kg

    # 磁場強度 1 Tesla
    B0 = 1.0  # [T]
    sim = ParticleTrajectorySimulation(
        field_type="ABC", q_over_m=q_over_m, particle_name="electron"
    )
    sim.magnetic_field.magnitude = B0

    # サイクロトロン周波数
    omega_c = PhysicalConstants.cyclotron_frequency(B0, q_over_m)
    T_c = 2 * np.pi / omega_c  # サイクロトロン周期
    print(f"\n電子パラメータ:")
    print(f"  質量: {PhysicalConstants.m_e:.3e} kg")
    print(f"  電荷: {PhysicalConstants.e:.3e} C")
    print(f"  比電荷: {q_over_m:.3e} C/kg")
    print(f"  サイクロトロン周期: {T_c:.3e} s ({T_c*1e9:.3f} ns)\n")

    # 初期条件: 位置 [m] と速度 [m/s]
    x0 = np.array([0.001, 0.0, 0.0])  # 1mm
    v0 = np.array([1e6, 0.0, 0.0])  # 1000 km/s (熱電子速度程度)

    # ラーモア半径
    r_L = PhysicalConstants.larmor_radius(np.linalg.norm(v0), B0, q_over_m)
    print(f"初期条件:")
    print(
        f"  速度: {np.linalg.norm(v0):.3e} m/s ({np.linalg.norm(v0)/PhysicalConstants.c:.4f}c)"
    )
    print(f"  ラーモア半径: {r_L:.3e} m ({r_L*1e3:.3f} mm)\n")

    initial_state = sim.set_initial_condition(x0, v0)

    # 数サイクロトロン周期分シミュレーション (適度なdt)
    n_cycles = 10
    t_max = n_cycles * T_c
    dt = T_c / 50  # 1周期を50点

    times, states = sim.run(initial_state, t_max=t_max, dt=dt)

    # 結果プロット
    sim.plot_trajectory(states)
    sim.plot_energy(times, states)


def example_single_particle():
    """単一電子の長時間追跡"""
    print("=" * 60)
    print("単一電子軌道計算例 - ABC磁場 (長時間)")
    print("=" * 60)

    # 電子パラメータ
    q_over_m = PhysicalConstants.q_over_m_electron
    B0 = 0.1  # 0.1 Tesla (弱めの磁場)

    sim = ParticleTrajectorySimulation(
        field_type="ABC", q_over_m=q_over_m, particle_name="electron"
    )
    sim.magnetic_field.magnitude = B0

    # 初期条件
    x0 = np.array([0.001, 0.0, 0.0])  # 1mm
    v0 = np.array([5e6, 0.0, 0.0])  # 5000 km/s
    initial_state = sim.set_initial_condition(x0, v0)

    # サイクロトロン周期
    omega_c = PhysicalConstants.cyclotron_frequency(B0, q_over_m)
    T_c = 2 * np.pi / omega_c

    # 50サイクル分
    n_cycles = 50
    t_max = n_cycles * T_c
    dt = T_c / 50

    times, states = sim.run(initial_state, t_max=t_max, dt=dt)

    # 結果プロット
    sim.plot_trajectory(states)
    sim.plot_energy(times, states)


def example_multiple_initial_conditions():
    """複数電子の軌道比較例"""
    print("=" * 60)
    print("複数電子の軌道比較 (8電子)")
    print("=" * 60)

    # 電子パラメータ
    q_over_m = PhysicalConstants.q_over_m_electron
    B0 = 0.1  # 0.1 Tesla

    sim = ParticleTrajectorySimulation(
        field_type="ABC", q_over_m=q_over_m, particle_name="electron"
    )
    sim.magnetic_field.magnitude = B0

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 異なる初期位置で計算
    initial_angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8個の異なる角度
    colors = plt.cm.hsv(np.linspace(0, 1, len(initial_angles)))

    # サイクロトロン周期
    omega_c = PhysicalConstants.cyclotron_frequency(B0, q_over_m)
    T_c = 2 * np.pi / omega_c
    n_cycles = 30
    t_max = n_cycles * T_c
    dt = T_c / 50

    for i, (angle, color) in enumerate(zip(initial_angles, colors)):
        radius = 0.001  # 1mm
        x0 = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        v0 = np.array([3e6, 0.0, 0.0])  # 3000 km/s
        initial_state = sim.set_initial_condition(x0, v0)

        times, states = sim.run(initial_state, t_max=t_max, dt=dt)

        ax.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            color=color,
            linewidth=0.6,
            label=f"θ={angle*180/np.pi:.0f}°",
            alpha=0.8,
        )
        ax.scatter(
            states[0, 0], states[0, 1], states[0, 2], c=[color], s=80, marker="o"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Multiple Particle Trajectories")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def example_different_fields():
    """異なる磁場での電子軌道比較"""
    print("=" * 60)
    print("異なる磁場での電子軌道比較")
    print("=" * 60)

    # 電子パラメータ
    q_over_m = PhysicalConstants.q_over_m_electron
    B0 = 0.05  # 0.05 Tesla

    field_types = ["ABC", "mirror"]
    fig = plt.figure(figsize=(16, 7))

    x0 = np.array([0.002, 0.0, 0.0])  # 2mm
    v0 = np.array([4e6, 1e6, 0.0])  # 4000, 1000, 0 km/s

    # サイクロトロン周期
    omega_c = PhysicalConstants.cyclotron_frequency(B0, q_over_m)
    T_c = 2 * np.pi / omega_c
    n_cycles = 40
    t_max = n_cycles * T_c
    dt = T_c / 50

    for i, field_type in enumerate(field_types):
        sim = ParticleTrajectorySimulation(
            field_type=field_type, q_over_m=q_over_m, particle_name="electron"
        )
        sim.magnetic_field.magnitude = B0
        initial_state = sim.set_initial_condition(x0, v0)

        times, states = sim.run(initial_state, t_max=t_max, dt=dt)

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(
            states[:, 0], states[:, 1], states[:, 2], "b-", linewidth=0.4, alpha=0.7
        )
        ax.scatter(
            states[0, 0],
            states[0, 1],
            states[0, 2],
            c="green",
            s=120,
            marker="o",
            label="Start",
            edgecolors="black",
        )
        ax.scatter(
            states[-1, 0],
            states[-1, 1],
            states[-1, 2],
            c="red",
            s=120,
            marker="x",
            label="End",
            linewidths=2,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{field_type} Field (t=0→150, {len(states)} points)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 実行例を選択
    print("\n磁場中の電子軌道シミュレーション")
    print("\n全て電子の実物理パラメータを使用")
    print("dtを荒くして高速計算\n")
    print("Examples:")
    print("  1: 基本電子軌道 (1T, 10サイクル)")
    print("  2: 長時間追跡 (0.1T, 50サイクル)")
    print("  3: 8電子同時追跡 (0.1T, 30サイクル)")
    print("  4: 異なる磁場比較 (0.05T, 40サイクル)")
    print("\n" + "=" * 60)

    example_electron()
    example_single_particle()
    example_multiple_initial_conditions()
    example_different_fields()
