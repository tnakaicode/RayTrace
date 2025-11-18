"""
ジャイロトロン共振器シミュレーション
Gyrotron Resonator Simulation

円筒共振器内での電子サイクロトロンメーザー(ECM)による
マイクロ波生成をシミュレートします。

物理モデル:
- 電子の螺旋運動 (ローレンツ力)
- TE モード電磁場との相互作用
- エネルギー交換 (電子 → 電場)
- 位相バンチング効果

使用例:
    python gyrotron_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Tuple, List
from scipy.special import jn, jn_zeros


class PhysicalConstants:
    """物理定数クラス"""

    # 基本定数
    c = 2.998e8  # 光速 [m/s]
    e = 1.602e-19  # 電気素量 [C]
    m_e = 9.109e-31  # 電子質量 [kg]
    m_p = 1.673e-27  # 陽子質量 [kg]
    epsilon_0 = 8.854e-12  # 真空の誘電率 [F/m]
    mu_0 = 1.257e-6  # 真空の透磁率 [H/m]

    # 電子の比電荷 [C/kg]
    q_over_m_electron = e / m_e  # ≈ 1.759e11 C/kg
    # 陽子の比電荷 [C/kg]
    q_over_m_proton = e / m_p  # ≈ 9.578e7 C/kg

    # 古典電子半径 r_0 = e^2/(4πε_0 m_e c^2)
    r_e = e**2 / (4 * np.pi * epsilon_0 * m_e * c**2)  # ≈ 2.818e-15 m

    @staticmethod
    def cyclotron_frequency(B: float, q_over_m: float) -> float:
        """サイクロトロン周波数 ω_c = qB/m [rad/s]"""
        return q_over_m * B

    @staticmethod
    def larmor_radius(v_perp: float, B: float, q_over_m: float) -> float:
        """ラーモア半径 r_L = v⊥/(ω_c) = m*v⊥/(qB) [m]"""
        omega_c = PhysicalConstants.cyclotron_frequency(B, q_over_m)
        return v_perp / omega_c if omega_c != 0 else np.inf

    @staticmethod
    def lorentz_factor(v: float) -> float:
        """ローレンツ因子 γ = 1/sqrt(1 - v²/c²)"""
        beta = v / PhysicalConstants.c
        if beta >= 1.0:
            return np.inf
        return 1.0 / np.sqrt(1.0 - beta**2)

    @staticmethod
    def lorentz_factor_from_energy(E_kin: float) -> float:
        """運動エネルギーからローレンツ因子 γ = 1 + E_kin/(m_e c²)"""
        return 1.0 + E_kin / (PhysicalConstants.m_e * PhysicalConstants.c**2)

    @staticmethod
    def relativistic_cyclotron_frequency(B: float, gamma: float) -> float:
        """相対論的サイクロトロン周波数 ω_c = eB/(γm_e)"""
        return PhysicalConstants.e * B / (gamma * PhysicalConstants.m_e)

    @staticmethod
    def synchrotron_power(
        gamma: float, v: np.ndarray, B: np.ndarray, q: float, m: float
    ) -> float:
        """
        シンクロトロン放射パワー [W]
        P = (2/3) * r_e * c * gamma^4 * (v × B)^2 / c^2
        または P = (q^4 B^2 gamma^2 v_perp^2) / (6πε_0 m^2 c^3)
        """
        v_cross_B = np.cross(v, B)
        v_perp_sq = np.sum(v_cross_B**2) / np.sum(B**2) if np.sum(B**2) > 0 else 0

        # 簡略化された式：電子の場合
        if abs(m - PhysicalConstants.m_e) / PhysicalConstants.m_e < 0.01:
            c = PhysicalConstants.c
            r_e = PhysicalConstants.r_e
            return (2.0 / 3.0) * r_e * c * gamma**4 * np.sum(v_cross_B**2) / c**2
        else:
            # 一般的な式
            c = PhysicalConstants.c
            eps0 = PhysicalConstants.epsilon_0
            B_mag = np.linalg.norm(B)
            return (q**4 * B_mag**2 * gamma**2 * v_perp_sq) / (
                6 * np.pi * eps0 * m**2 * c**3
            )

    @staticmethod
    def radiation_damping_force(
        gamma: float, v: np.ndarray, B: np.ndarray, q: float, m: float
    ) -> np.ndarray:
        """
        放射減衰力 (Abraham-Lorentz-Dirac力の簡略形)
        F_rad = -P/v * (v/|v|)  速度と逆方向
        """
        v_mag = np.linalg.norm(v)
        if v_mag < 1e-10:
            return np.zeros(3)

        P = PhysicalConstants.synchrotron_power(gamma, v, B, q, m)
        return -P / v_mag * (v / v_mag)


class CylindricalCavity:
    """円筒共振器クラス"""

    def __init__(self, radius: float, length: float, mode_m: int = 1, mode_n: int = 1):
        """
        Parameters:
            radius: 共振器半径 [m]
            length: 共振器長さ [m]
            mode_m: 方位角モード数
            mode_n: 径方向モード数
        """
        self.radius = radius
        self.length = length
        self.mode_m = mode_m
        self.mode_n = mode_n

        # TE_{mn} モードのカットオフ波数
        self.nu_mn = jn_zeros(mode_m, mode_n)[-1] / radius

        # 共振周波数 (簡略化: 軸方向定在波なし)
        self.f_resonance = PhysicalConstants.c * self.nu_mn / (2 * np.pi)
        self.omega_resonance = 2 * np.pi * self.f_resonance

        print(f"共振器パラメータ:")
        print(f"  半径: {radius*1e2:.2f} cm")
        print(f"  長さ: {length*1e2:.2f} cm")
        print(f"  モード: TE_{mode_m}{mode_n}")
        print(f"  共振周波数: {self.f_resonance*1e-9:.2f} GHz")

    def get_rf_fields(
        self, r: float, phi: float, z: float, t: float, E_amplitude: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RF電磁場を計算 (TE_{mn} モード)

        Parameters:
            r, phi, z: 円筒座標
            t: 時刻
            E_amplitude: 電場振幅 [V/m]

        Returns:
            E_field: 電場ベクトル (円筒座標) [V/m]
            B_field: 磁場ベクトル (円筒座標) [T]
        """
        m = self.mode_m
        nu = self.nu_mn
        omega = self.omega_resonance

        # 境界条件チェック
        if r > self.radius or z < 0 or z > self.length:
            return np.zeros(3), np.zeros(3)

        # ベッセル関数とその微分
        x = nu * r
        J_m = jn(m, x)
        # ベッセル関数の微分: J_m'(x) = (J_{m-1}(x) - J_{m+1}(x))/2
        J_m_prime = (jn(m - 1, x) - jn(m + 1, x)) / 2.0 if x > 0 else 0.0

        # 時間・位相因子
        phase = omega * t - m * phi
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        # 電場成分 (円筒座標)
        E_r = -E_amplitude * (m / (nu * r)) * J_m * sin_phase
        E_phi = E_amplitude * J_m_prime * cos_phase
        E_z = 0.0  # TEモードなのでE_z = 0

        # 磁場成分 (マクスウェル方程式から)
        B_r = 0.0
        B_phi = 0.0
        B_z = E_amplitude * nu * J_m * cos_phase / PhysicalConstants.c

        E_field = np.array([E_r, E_phi, E_z])
        B_field = np.array([B_r, B_phi, B_z])

        return E_field, B_field


class GyrotronElectronBeam:
    """ジャイロトロン電子ビームクラス"""

    def __init__(
        self,
        n_electrons: int,
        B0: float,
        V_beam: float,
        alpha: float,
        radius_beam: float,
    ):
        """
        Parameters:
            n_electrons: 電子数
            B0: 静磁場強度 [T]
            V_beam: ビーム電圧 [V]
            alpha: ピッチ角 [rad]
            radius_beam: ビーム半径 [m]
        """
        self.n_electrons = n_electrons
        self.B0 = B0
        self.V_beam = V_beam
        self.alpha = alpha
        self.radius_beam = radius_beam

        # 相対論的エネルギー計算
        E_kin = PhysicalConstants.e * V_beam  # 運動エネルギー [J]
        self.gamma = PhysicalConstants.lorentz_factor_from_energy(E_kin)

        # 相対論的サイクロトロン周波数
        self.omega_c = PhysicalConstants.relativistic_cyclotron_frequency(
            B0, self.gamma
        )
        self.f_c = self.omega_c / (2 * np.pi)

        # ビームエネルギーから速度を計算 (相対論的)
        # E_kin = (γ-1)m_e c² → v = c√(1 - 1/γ²)
        beta = np.sqrt(1.0 - 1.0 / self.gamma**2)
        self.v_total = beta * PhysicalConstants.c
        self.v_perp = self.v_total * np.sin(alpha)  # 垂直速度
        self.v_parallel = self.v_total * np.cos(alpha)  # 平行速度

        # 相対論的ラーモア半径
        self.r_L = (
            self.gamma
            * PhysicalConstants.m_e
            * self.v_perp
            / (PhysicalConstants.e * B0)
        )

        print(f"\n電子ビームパラメータ:")
        print(f"  電子数: {n_electrons}")
        print(f"  ビーム電圧: {V_beam*1e-3:.1f} kV")
        print(f"  静磁場: {B0:.2f} T")
        print(f"  ローレンツ因子: γ = {self.gamma:.4f}")
        print(f"  相対論因子: β = v/c = {self.v_total/PhysicalConstants.c:.4f}")
        print(f"  サイクロトロン周波数: {self.f_c*1e-9:.2f} GHz (相対論的補正済)")
        print(f"  ピッチ角: {np.degrees(alpha):.1f}°")
        print(
            f"  全速度: {self.v_total*1e-6:.2f} × 10^6 m/s ({self.v_total/PhysicalConstants.c*100:.1f}% of c)"
        )
        print(f"  垂直速度: {self.v_perp*1e-6:.2f} × 10^6 m/s")
        print(f"  ラーモア半径: {self.r_L*1e3:.2f} mm (相対論的)")

    def initialize_electrons(self) -> np.ndarray:
        """
        電子の初期位置・速度を設定

        Returns:
            states: [n_electrons, 6] 配列 (x, y, z, vx, vy, vz)
        """
        states = np.zeros((self.n_electrons, 6))

        # 初期位置: ビーム半径内にランダム配置
        for i in range(self.n_electrons):
            # 円周上に等間隔配置
            phi = 2 * np.pi * i / self.n_electrons
            r = self.radius_beam

            states[i, 0] = r * np.cos(phi)  # x
            states[i, 1] = r * np.sin(phi)  # y
            states[i, 2] = 0.0  # z (入口)

            # 初期速度: 螺旋運動
            states[i, 3] = -self.v_perp * np.sin(phi)  # vx
            states[i, 4] = self.v_perp * np.cos(phi)  # vy
            states[i, 5] = self.v_parallel  # vz

        return states


class GyrotronSimulator:
    """ジャイロトロンシミュレータ"""

    def __init__(self, cavity: CylindricalCavity, beam: GyrotronElectronBeam):
        """
        Parameters:
            cavity: 共振器
            beam: 電子ビーム
        """
        self.cavity = cavity
        self.beam = beam
        self.E_amplitude = 0.0  # RF電場振幅 (初期値)
        self.energy_extracted = 0.0  # 取り出しエネルギー

    def cartesian_to_cylindrical(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """デカルト座標 → 円筒座標"""
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return r, phi, z

    def cylindrical_to_cartesian_vector(
        self, V_r: float, V_phi: float, V_z: float, phi: float
    ) -> np.ndarray:
        """円筒座標ベクトル → デカルト座標ベクトル"""
        Vx = V_r * np.cos(phi) - V_phi * np.sin(phi)
        Vy = V_r * np.sin(phi) + V_phi * np.cos(phi)
        Vz = V_z
        return np.array([Vx, Vy, Vz])

    def equation_of_motion(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        運動方程式

        Parameters:
            state: [x, y, z, vx, vy, vz]
            t: 時刻

        Returns:
            dstate/dt
        """
        x, y, z, vx, vy, vz = state

        # 円筒座標に変換
        r, phi, z_cyl = self.cartesian_to_cylindrical(x, y, z)

        # 静磁場 (z方向)
        B_static = np.array([0, 0, self.beam.B0])

        # RF電磁場
        E_rf_cyl, B_rf_cyl = self.cavity.get_rf_fields(r, phi, z, t, self.E_amplitude)
        E_rf = self.cylindrical_to_cartesian_vector(
            E_rf_cyl[0], E_rf_cyl[1], E_rf_cyl[2], phi
        )
        B_rf = self.cylindrical_to_cartesian_vector(
            B_rf_cyl[0], B_rf_cyl[1], B_rf_cyl[2], phi
        )

        # 全磁場
        B_total = B_static + B_rf

        # 相対論的ローレンツ力
        v = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v)
        gamma_instant = PhysicalConstants.lorentz_factor(v_mag)

        # 相対論的運動方程式: dp/dt = q(E + v×B), p = γmv
        # dv/dt = (q/γm)(E + v×B) - (γ̇/γ)v
        # 簡略化: γが大きく変化しない場合
        q_over_m_rel = PhysicalConstants.e / (gamma_instant * PhysicalConstants.m_e)

        acceleration = q_over_m_rel * (E_rf + np.cross(v, B_total))

        dstate_dt = np.zeros(6)
        dstate_dt[0:3] = v
        dstate_dt[3:6] = acceleration

        return dstate_dt

    def rk4_step(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        """4次ルンゲ・クッタ法"""
        k1 = dt * self.equation_of_motion(state, t)
        k2 = dt * self.equation_of_motion(state + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * self.equation_of_motion(state + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * self.equation_of_motion(state + k3, t + dt)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def simulate(
        self, t_max: float, dt: float, E_initial: float = 1e3
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        シミュレーション実行

        Parameters:
            t_max: 最大時間 [s]
            dt: 時間刻み [s]
            E_initial: 初期RF電場振幅 [V/m]

        Returns:
            times: 時刻配列
            electron_trajectories: 各電子の軌跡リスト
        """
        self.E_amplitude = E_initial

        n_steps = int(t_max / dt) + 1
        times = np.linspace(0, t_max, n_steps)

        # 電子初期化
        states = self.beam.initialize_electrons()

        # 軌跡保存
        electron_trajectories = [
            np.zeros((n_steps, 6)) for _ in range(self.beam.n_electrons)
        ]
        for i in range(self.beam.n_electrons):
            electron_trajectories[i][0] = states[i]

        print(f"\nシミュレーション開始...")
        print(f"  時間ステップ: {n_steps}")
        print(f"  dt: {dt*1e12:.2f} ps")
        print(f"  全時間: {t_max*1e9:.2f} ns")

        # 時間発展
        for step in range(1, n_steps):
            if step % (n_steps // 10) == 0:
                print(f"  進行: {step/n_steps*100:.0f}%")

            for i in range(self.beam.n_electrons):
                states[i] = self.rk4_step(states[i], times[step - 1], dt)
                electron_trajectories[i][step] = states[i]

            # 簡易的なRF電場増幅モデル (実際はより複雑)
            # 電子からエネルギーを受け取ってRF場が成長
            if step % 100 == 0:
                self.E_amplitude *= 1.001  # 緩やかな成長

        print("シミュレーション完了!")

        return times, electron_trajectories

    def calculate_field_distribution(
        self, z_plane: float, n_grid: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RF電場分布を計算

        Parameters:
            z_plane: z座標平面
            n_grid: グリッド点数

        Returns:
            X, Y, E_magnitude: メッシュグリッドと電場強度
        """
        x = np.linspace(-self.cavity.radius, self.cavity.radius, n_grid)
        y = np.linspace(-self.cavity.radius, self.cavity.radius, n_grid)
        X, Y = np.meshgrid(x, y)

        E_magnitude = np.zeros_like(X)

        for i in range(n_grid):
            for j in range(n_grid):
                r, phi, z = self.cartesian_to_cylindrical(X[i, j], Y[i, j], z_plane)

                if r <= self.cavity.radius:
                    E_rf_cyl, _ = self.cavity.get_rf_fields(
                        r, phi, z, 0, self.E_amplitude
                    )
                    E_magnitude[i, j] = np.linalg.norm(E_rf_cyl)

        return X, Y, E_magnitude


def plot_results(
    simulator: GyrotronSimulator, times: np.ndarray, trajectories: List[np.ndarray]
):
    """結果の可視化"""

    # 図1: 3D電子軌道
    fig = plt.figure(figsize=(18, 6))

    # 3D軌道プロット
    ax1 = fig.add_subplot(131, projection="3d")

    # 数本の電子軌道を描画
    n_plot = min(5, len(trajectories))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_plot))

    for i in range(n_plot):
        traj = trajectories[i]
        ax1.plot(
            traj[:, 0] * 1e3,
            traj[:, 1] * 1e3,
            traj[:, 2] * 1e2,
            color=colors[i],
            linewidth=1,
            alpha=0.7,
            label=f"Electron {i+1}",
        )

    # 共振器の円筒を描画
    theta = np.linspace(0, 2 * np.pi, 50)
    z_cyl = np.linspace(0, simulator.cavity.length * 1e2, 2)
    Theta, Z = np.meshgrid(theta, z_cyl)
    X_cyl = simulator.cavity.radius * 1e3 * np.cos(Theta)
    Y_cyl = simulator.cavity.radius * 1e3 * np.sin(Theta)
    ax1.plot_surface(X_cyl, Y_cyl, Z, alpha=0.1, color="gray")

    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (cm)")
    ax1.set_title("Electron Trajectories in Gyrotron Cavity")
    ax1.legend(fontsize=8)

    # XY平面投影
    ax2 = fig.add_subplot(132)
    for i in range(n_plot):
        traj = trajectories[i]
        ax2.plot(
            traj[:, 0] * 1e3, traj[:, 1] * 1e3, color=colors[i], linewidth=1, alpha=0.7
        )

    circle = plt.Circle(
        (0, 0),
        simulator.cavity.radius * 1e3,
        fill=False,
        color="gray",
        linestyle="--",
        linewidth=2,
    )
    ax2.add_patch(circle)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("XY Projection")
    ax2.axis("equal")
    ax2.grid(True)

    # RF電場分布 (contourf)
    ax3 = fig.add_subplot(133)
    z_plane = simulator.cavity.length / 2  # 中央断面
    X, Y, E_mag = simulator.calculate_field_distribution(z_plane, n_grid=100)

    contour = ax3.contourf(X * 1e3, Y * 1e3, E_mag, levels=20, cmap="hot")
    ax3.contour(
        X * 1e3, Y * 1e3, E_mag, levels=10, colors="black", linewidths=0.5, alpha=0.3
    )
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label("Electric Field (V/m)", rotation=270, labelpad=20)

    circle = plt.Circle(
        (0, 0),
        simulator.cavity.radius * 1e3,
        fill=False,
        color="white",
        linestyle="--",
        linewidth=2,
    )
    ax3.add_patch(circle)

    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_title(
        f"RF E-field Distribution (TE$_{{{simulator.cavity.mode_m}{simulator.cavity.mode_n}}}$ mode)"
    )
    ax3.axis("equal")

    plt.tight_layout()
    plt.show()

    # 図2: エネルギー時間発展
    fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(10, 8))

    # 平均運動エネルギー（相対論的）
    n_electrons = len(trajectories)
    avg_energy = np.zeros(len(times))

    for i in range(n_electrons):
        traj = trajectories[i]
        v = np.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2 + traj[:, 5] ** 2)

        # 相対論的運動エネルギー: E_kin = (γ - 1)m_e c²
        gamma_traj = np.array(
            [PhysicalConstants.lorentz_factor(v_i) if v_i > 0 else 1.0 for v_i in v]
        )
        kinetic_energy = (
            (gamma_traj - 1.0) * PhysicalConstants.m_e * PhysicalConstants.c**2
        )
        avg_energy += kinetic_energy / n_electrons

    ax4.plot(times * 1e9, avg_energy, "b-", linewidth=2)
    ax4.set_xlabel("Time (ns)")
    ax4.set_ylabel("Average Kinetic Energy (J)")
    ax4.set_title("Electron Beam Energy Evolution")
    ax4.grid(True)

    # エネルギー変化率 (パワー抽出)
    energy_loss = (avg_energy[0] - avg_energy) / avg_energy[0] * 100
    ax5.plot(times * 1e9, energy_loss, "r-", linewidth=2)
    ax5.set_xlabel("Time (ns)")
    ax5.set_ylabel("Energy Loss (%)")
    ax5.set_title("Energy Transfer to RF Field")
    ax5.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """メイン実行関数"""
    print("=" * 60)
    print("ジャイロトロン共振器シミュレーション")
    print("=" * 60)

    # パラメータ設定
    # 共振器 (140 GHz帯を想定)
    cavity_radius = 0.02  # 2 cm
    cavity_length = 0.10  # 10 cm
    mode_m = 10  # TE_{31} モードなど
    mode_n = 11

    cavity = CylindricalCavity(cavity_radius, cavity_length, mode_m, mode_n)

    # 電子ビーム
    n_electrons = 20  # 計算コスト削減のため少数
    B0 = 5.0  # 5 Tesla
    V_beam = 80e3  # 80 kV
    pitch_angle = np.radians(1.2)  # 60度
    beam_radius = 0.010  # 5 mm

    beam = GyrotronElectronBeam(n_electrons, B0, V_beam, pitch_angle, beam_radius)

    # シミュレータ
    simulator = GyrotronSimulator(cavity, beam)

    # シミュレーション実行
    # サイクロトロン周期
    T_c = 2 * np.pi / beam.omega_c
    n_cycles = 10
    t_max = n_cycles * T_c
    dt = T_c / 1000  # 1周期を100点でサンプル

    times, trajectories = simulator.simulate(t_max, dt, E_initial=5e3)

    # 結果プロット
    plot_results(simulator, times, trajectories)
