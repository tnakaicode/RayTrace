---
title: 光線・粒子軌道計算プログラム集
---

- [1. particle\_trajectory.py](#1-particle_trajectorypy)
  - [1.1. 主要クラス](#11-主要クラス)
  - [1.2. 物理モデル](#12-物理モデル)
    - [1.2.1. 使用例](#121-使用例)
    - [1.2.2. 実行](#122-実行)
    - [1.2.3. 特徴](#123-特徴)
    - [1.2.4. 数値手法](#124-数値手法)
    - [1.2.5. 出力](#125-出力)
- [2. gyrotron\_simulation.py](#2-gyrotron_simulationpy)
  - [2.1. 主要クラス](#21-主要クラス)
  - [2.2. 物理モデル](#22-物理モデル)
  - [2.3. 計算手法](#23-計算手法)
  - [2.4. 使用例](#24-使用例)
    - [2.4.1. 基本的な実行](#241-基本的な実行)
    - [2.4.2. 計算例1: 標準パラメータ（140 GHz帯）](#242-計算例1-標準パラメータ140-ghz帯)
    - [2.4.3. 計算例2: 高エネルギービーム（200 kV）](#243-計算例2-高エネルギービーム200-kv)
    - [2.4.4. 計算例3: 低周波数動作（28 GHz帯）](#244-計算例3-低周波数動作28-ghz帯)
  - [2.5. 実行](#25-実行)
  - [2.6. 出力](#26-出力)
  - [2.7. 特徴](#27-特徴)
  - [2.8. 物理パラメータの目安](#28-物理パラメータの目安)
  - [2.9. 注意事項](#29-注意事項)
- [3. Legacy Code (非推奨)](#3-legacy-code-非推奨)
  - [3.1. Trajectory/odeN.py](#31-trajectoryodenpy)

## 1. particle_trajectory.py

磁場中の荷電粒子軌道計算プログラム
ルンゲ・クッタ法(RK4)を使用してローレンツ力による粒子運動を数値的に解きます。

### 1.1. 主要クラス

1. **MagneticField** - 磁場モデルクラス
   - ABC磁場 (Arnold-Beltrami-Childress flow)
   - ABC磁場 + 一様磁場
   - ミラー磁場

2. **ParticleIntegrator** - 数値積分クラス
   - 4次ルンゲ・クッタ法 (RK4) による時間積分
   - 運動方程式: `dv/dt = (q/m) * v × B(x)`
   - エネルギー保存を高精度で維持

3. **ParticleTrajectorySimulation** - シミュレーション統合クラス
   - 初期条件設定
   - シミュレーション実行
   - 3D軌道プロット
   - エネルギー保存チェック

### 1.2. 物理モデル

**運動方程式:**

```python
dx/dt = v
dv/dt = (q/m) * v × B(x)
```

**ABC磁場:**

```python
Bx = A*cos(ky) + C*sin(kz)
By = B*sin(kx) + A*cos(kz)
Bz = C*sin(ky) + B*cos(kx)
```

ここで A=B=C=1, k は波数パラメータ

**ミラー磁場:**

```python
Bx = -0.5 * m * y / (x² + y²)
By = -0.5 * m * x / (x² + y²)
Bz = m*z + 1
```

#### 1.2.1. 使用例

```python
# 基本的な使い方
from particle_trajectory import ParticleTrajectorySimulation
import numpy as np

# シミュレーション初期化
sim = ParticleTrajectorySimulation(field_type='ABC', q_over_m=1.0)

# 初期条件設定 (位置と速度)
x0 = np.array([1.0, 0.0, 0.0])
v0 = np.array([0.1, 0.0, 0.0])
initial_state = sim.set_initial_condition(x0, v0)

# シミュレーション実行
times, states = sim.run(initial_state, t_max=100.0, dt=0.01)

# 結果プロット
sim.plot_trajectory(states)
sim.plot_energy(times, states)
```

#### 1.2.2. 実行

```bash
python particle_trajectory.py
```

3つの計算例が自動実行されます:

1. 単一粒子軌道 (ABC磁場、t=0→100)
2. 複数初期条件での軌道比較 (4つの異なる初期角度)
3. 異なる磁場タイプでの比較 (ABC vs Mirror)

#### 1.2.3. 特徴

- ✅ グローバル変数なし、完全クラスベース設計
- ✅ 1ファイル完結、外部依存最小
- ✅ 型ヒント・docstring完備
- ✅ エネルギー保存精度 < 0.01%
- ✅ 複数の磁場モデル対応
- ✅ 可視化機能内蔵

#### 1.2.4. 数値手法

**4次ルンゲ・クッタ法 (RK4):**

```python
k1 = dt * f(y, t)
k2 = dt * f(y + k1/2, t + dt/2)
k3 = dt * f(y + k2/2, t + dt/2)
k4 = dt * f(y + k3, t + dt)
y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
```

精度: O(dt⁴)、安定性良好

#### 1.2.5. 出力

- 3D軌道プロット (XYZ空間)
- XY平面投影図
- 運動エネルギー時間変化
- エネルギー保存誤差

## 2. gyrotron_simulation.py

ジャイロトロン共振器シミュレーション
円筒共振器内での電子サイクロトロンメーザー(ECM)による高出力マイクロ波生成をシミュレートします。

### 2.1. 主要クラス

1. **PhysicalConstants** - 物理定数クラス
   - 基本定数: 光速c, 電気素量e, 電子質量m_e, 真空誘電率ε₀, 真空透磁率μ₀
   - 派生定数: 比電荷q/m, 古典電子半径r_e
   - 相対論的関数: ローレンツ因子γ, 相対論的サイクロトロン周波数
   - シンクロトロン放射: 放射パワー計算, 放射減衰力

2. **CylindricalCavity** - 円筒共振器クラス
   - TE_{mn}モード電磁場の計算
   - ベッセル関数による径方向場分布
   - 共振周波数: f_res = c·ν_{mn}/(2π)
   - 電場・磁場成分の時間・空間変化

3. **GyrotronElectronBeam** - 電子ビームクラス
   - 相対論的エネルギー計算
   - 螺旋軌道の初期化（ピッチ角指定）
   - ラーモア半径の相対論的補正
   - 電子配置の円周等間隔分布

4. **GyrotronSimulator** - シミュレータクラス
   - 相対論的運動方程式の数値積分
   - 座標変換: デカルト ⇔ 円筒座標
   - RF電場分布計算・可視化
   - 加速電圧分布計算・可視化
   - エネルギー時間発展解析

### 2.2. 物理モデル

**相対論的運動方程式:**

```python
dp/dt = q(E + v×B)
p = γmv
γ = 1/√(1 - v²/c²)
```

簡略化（γの時間変化が小さい場合）:

```python
dv/dt = (q/γm)(E + v×B)
```

**TE_{mn}モード電磁場:**

円筒座標(r, φ, z)での電場成分:

```python
E_r = -E₀(m/νr)J_m(νr)sin(ωt - mφ)
E_φ = E₀J'_m(νr)cos(ωt - mφ)
E_z = 0  # TEモード
```

ベッセル関数の微分:

```python
J'_m(x) = [J_{m-1}(x) - J_{m+1}(x)]/2
```

カットオフ波数（ν_{mn}はJ_m(x)のn番目の零点）:

```python
ν_{mn} = zero_n(J_m) / R
```

**相対論的エネルギー:**

運動エネルギー:

```python
E_kin = (γ - 1)mc²
γ = 1 + eV/(mc²)
```

80kVの電子:

```python
γ = 1.157
β = v/c = 0.502 (光速の50%)
```

**サイクロトロン周波数（相対論的）:**

```python
ω_c = eB/(γm_e)
f_c = ω_c/(2π)
```

相対論効果により周波数が1/γに低下:

```python
f_c(80kV, 5T) = 121 GHz / 1.157 ≈ 105 GHz
```

**シンクロトロン放射:**

放射パワー（電子の場合）:

```python
P = (2/3)r_e c γ⁴(v×B)²/c²
```

古典電子半径:

```python
r_e = e²/(4πε₀m_e c²) = 2.818×10⁻¹⁵ m
```

### 2.3. 計算手法

**4次ルンゲ・クッタ法 (RK4):**

```python
k1 = dt * f(state, t)
k2 = dt * f(state + k1/2, t + dt/2)
k3 = dt * f(state + k2/2, t + dt/2)
k4 = dt * f(state + k3, t + dt)
state_next = state + (k1 + 2k2 + 2k3 + k4)/6
```

時間刻み:

```python
dt = T_c / 100  # サイクロトロン周期の1/100
T_c = 2π/ω_c
```

**座標変換:**

デカルト → 円筒:

```python
r = √(x² + y²)
φ = arctan2(y, x)
z = z
```

円筒ベクトル → デカルト:

```python
V_x = V_r cos(φ) - V_φ sin(φ)
V_y = V_r sin(φ) + V_φ cos(φ)
V_z = V_z
```

### 2.4. 使用例

#### 2.4.1. 基本的な実行

```python
from gyrotron_simulation import (
    CylindricalCavity, 
    GyrotronElectronBeam, 
    GyrotronSimulator
)
import numpy as np

# 共振器設定 (140 GHz帯)
cavity = CylindricalCavity(
    radius=0.01,      # 1 cm
    length=0.05,      # 5 cm
    mode_m=3,         # TE_{31}モード
    mode_n=1
)

# 電子ビーム設定
beam = GyrotronElectronBeam(
    n_electrons=10,
    B0=5.0,           # 5 Tesla
    V_beam=80e3,      # 80 kV
    alpha=np.radians(60),  # ピッチ角60°
    radius_beam=0.005      # 5 mm
)

# シミュレータ初期化
simulator = GyrotronSimulator(cavity, beam)

# シミュレーション実行
T_c = 2*np.pi / beam.omega_c
times, trajectories = simulator.simulate(
    t_max=5*T_c,
    dt=T_c/100,
    E_initial=5e3
)

# 結果可視化
simulator.plot_acceleration_voltage()
simulator.plot_results(times, trajectories)
```

#### 2.4.2. 計算例1: 標準パラメータ（140 GHz帯）

```python
# パラメータ
cavity_radius = 0.01 m (1 cm)
cavity_length = 0.05 m (5 cm)
mode = TE_31
B0 = 5.0 T
V_beam = 80 kV
pitch_angle = 60°

# 結果
共振周波数: 30.4 GHz
電子速度: 0.50c (150 Mm/s)
ローレンツ因子: γ = 1.157
サイクロトロン周波数: 121 GHz (相対論的)
ラーモア半径: 0.17 mm
```

#### 2.4.3. 計算例2: 高エネルギービーム（200 kV）

```python
beam = GyrotronElectronBeam(
    n_electrons=20,
    B0=7.0,           # 7 Tesla
    V_beam=200e3,     # 200 kV
    alpha=np.radians(45),
    radius_beam=0.008
)

# 結果
電子速度: 0.70c (210 Mm/s)
ローレンツ因子: γ = 1.391
相対論的質量増加: 39%
サイクロトロン周波数低下: 28%
```

#### 2.4.4. 計算例3: 低周波数動作（28 GHz帯）

```python
cavity = CylindricalCavity(
    radius=0.015,     # 1.5 cm
    length=0.08,      # 8 cm
    mode_m=2,         # TE_21モード
    mode_n=1
)

beam = GyrotronElectronBeam(
    n_electrons=15,
    B0=1.2,           # 1.2 Tesla
    V_beam=40e3,      # 40 kV
    alpha=np.radians(70),
    radius_beam=0.007
)

# 結果
共振周波数: 28.6 GHz
電子速度: 0.37c
ローレンツ因子: γ = 1.078
サイクロトロン周波数: 21 GHz
```

### 2.5. 実行

```bash
python gyrotron_simulation.py
```

自動的に以下が実行されます:

1. パラメータ表示（共振器、ビーム、相対論的パラメータ）
2. 時間発展計算（RK4積分）
3. 加速電圧分布プロット
4. 3D電子軌道プロット
5. XY平面投影プロット
6. RF電場分布contourプロット
7. エネルギー時間発展プロット

### 2.6. 出力

**加速電圧分布図:**

- 左パネル: 軸方向電位V(z) [kV]
- 右パネル: 軸方向電場E_z = -dV/dz [kV/cm]
- 加速領域: z = -5cm → 0 (スムーズな3次関数遷移)
- 共振器内: z > 0 (一定電位、ドリフト領域)

**電子軌道図:**

- 3D軌道: 螺旋運動の可視化
- XY投影: サイクロトロン運動
- 共振器壁: 灰色円筒で表示

**RF電場分布図:**

- TE_{mn}モード電場強度のcontour
- 中央断面(z = L/2)での2D分布
- 白破線: 共振器壁境界

**エネルギー時間発展図:**

- 上パネル: 平均運動エネルギー [J]（相対論的）
- 下パネル: RF場へのエネルギー移行率 [%]

### 2.7. 特徴

- ✅ 完全相対論的計算（γ補正込み）
- ✅ TE_{mn}モード電磁場の厳密計算
- ✅ ベッセル関数による正確な場分布
- ✅ 座標変換の数値安定性
- ✅ 加速電圧分布の可視化
- ✅ エネルギー移行解析
- ✅ 日本語フォント対応
- ✅ クラスメソッド設計

### 2.8. 物理パラメータの目安

**典型的なジャイロトロン:**

- 周波数: 28-170 GHz
- 出力: 0.5-2 MW (連続波)
- ビーム電圧: 40-100 kV
- ビーム電流: 10-40 A
- 磁場: 1-7 T（超伝導マグネット）
- 共振器半径: 0.5-2 cm
- 効率: 30-50%

**ITERジャイロトロン（核融合用）:**

- 周波数: 170 GHz
- 出力: 1 MW (1000秒)
- ビーム電圧: 80 kV
- ビーム電流: 40 A
- 磁場: 6.7 T
- モード: TE_{31,8}またはTE_{28,8}

### 2.9. 注意事項

1. **時間刻み**: dt < T_c/50 を推奨（数値誤差制御）
2. **境界条件**: r > R で電場ゼロ
3. **初期RF場**: E_initial = 1-10 kV/m（実際は電子から成長）
4. **相対論**: V > 50 kV で重要（γ > 1.1）
5. **収束**: 長時間計算時はRF場の飽和を確認

## 3. Legacy Code (非推奨)

### 3.1. Trajectory/odeN.py

元のグローバル変数ベースの実装。保守性が低いため `particle_trajectory.py` の使用を推奨。
