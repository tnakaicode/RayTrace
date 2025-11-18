---
title: 光線・粒子軌道計算プログラム集
---
## particle_trajectory.py

磁場中の荷電粒子軌道計算プログラム
ルンゲ・クッタ法(RK4)を使用してローレンツ力による粒子運動を数値的に解きます。

### 主要クラス

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

### 物理モデル

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

#### 使用例

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

#### 実行

```bash
python particle_trajectory.py
```

3つの計算例が自動実行されます:

1. 単一粒子軌道 (ABC磁場、t=0→100)
2. 複数初期条件での軌道比較 (4つの異なる初期角度)
3. 異なる磁場タイプでの比較 (ABC vs Mirror)

#### 特徴

- ✅ グローバル変数なし、完全クラスベース設計
- ✅ 1ファイル完結、外部依存最小
- ✅ 型ヒント・docstring完備
- ✅ エネルギー保存精度 < 0.01%
- ✅ 複数の磁場モデル対応
- ✅ 可視化機能内蔵

#### 数値手法

**4次ルンゲ・クッタ法 (RK4):**

```python
k1 = dt * f(y, t)
k2 = dt * f(y + k1/2, t + dt/2)
k3 = dt * f(y + k2/2, t + dt/2)
k4 = dt * f(y + k3, t + dt)
y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
```

精度: O(dt⁴)、安定性良好

#### 出力

- 3D軌道プロット (XYZ空間)
- XY平面投影図
- 運動エネルギー時間変化
- エネルギー保存誤差

## Legacy Code (非推奨)

### Trajectory/odeN.py

元のグローバル変数ベースの実装。保守性が低いため `particle_trajectory.py` の使用を推奨。
