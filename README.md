# wtclean

风电机组 SCADA 功率曲线数据清洗工具：用「迭代分箱 + MAD（中位数绝对偏差）」把**正常运行数据**从原始 SCADA 数据中初筛出来。

## 这个包是做什么的

输入**一台风机**的 SCADA 数据（至少要含风速、有功功率两列，通常为 10 分钟统计记录）。

`PowerCurveFiltering.process()` 会把数据按行切分成两份返回：

- **`normal_df`**：风机**正常运行**的数据点——功率落在该风速区间应有的功率带内。**后续做功率曲线建模、工况分析、机器学习等时主要用这一份**。
- **`abnormal_df`**：其余所有被剔除的点。⚠️ **它不是"严格异常集"**，而是混了多种情况：停机、故障、限电/降载、桨距控制等**其它工况**、以及真正的传感器/数据异常。想区分异常类型，需要后续用更高阶方法（如 GAM/sigmoid 拟合、时间维度分析）处理。

单台风机的整条清洗流水线分三步：

1. **去停机**：风速 ≥ 切入风速、但功率 ≤ 1 kW 的点（风机已停机）→ 剔除。
2. **高风低功率粗洗**：风速 ≥ 额定风速、但功率 < `low_power_ratio × 额定功率` 的点（限电、降载、桨距/故障等非正常发电）→ 剔除。
3. **迭代 bin+MAD 精洗**：按 `bin_interval` 把风速分箱，对每箱算功率中位数与 MAD，箱内功率落在 `中位数 ± z_coeff × MAD` 带外的点 → 剔除。剔除后**重新分箱再剔**，重复 `filter_cycle` 轮（每轮带会随数据变干净而收窄）。

几个内置的判定约定（避免误杀正常点）：

- 箱内样本 < 3，或功率离散度为 0（典型如额定功率平台），该箱**不判定、直接保留**；
- 风速低于切入风速的点默认保留（近零功率属正常待机）；
- 恒定功率的"水平聚集"类异常（如某段时间被固定在某一功率）**bin+MAD 识别不出来**，这是本方法的能力上限，应交给后续更精细的方案。

## 安装

```bash
git clone <this-repo>
cd wtclean
pip install -r requirements.txt
pip install .
```

## 快速开始（单台风机）

```python
import pandas as pd
from wtclean import PowerCurveFiltering

df = pd.read_csv("data.csv")
turbine = df[df["Wind_turbine_name"] == "R80721"]   # 取单台风机

pc_filter = PowerCurveFiltering(
    windspeed_label="Ws_avg",     # 风速列名
    power_label="P_avg",          # 有功功率列名
    df=turbine,                   # 单台风机数据
    cut_in_speed=3.5,             # 切入风速（厂商参数，必填）
    rated_wind_speed=14.5,        # 额定风速（厂商参数，必填）
    rated_power=2050,             # 额定功率 kW（厂商参数，必填）
    low_power_ratio=0.9,          # 高风低功率粗洗阈值（默认 0.9）
    bin_interval=0.5,             # 风速分箱宽度（默认 0.5）
    z_coeff=2.5,                  # 正常带宽度系数（默认 2.5）
    filter_cycle=3,               # 迭代轮数（默认 3）
    return_fig=False,             # 是否保存功率曲线图
    image_path="",                # 出图时的完整输出文件路径
)

normal_df, abnormal_df = pc_filter.process()
```

运行时会逐轮打印删除情况，例如：

```
iteration 1: removed 3864 (7.21%), remaining 49724
iteration 2: removed 1655 (3.33%), remaining 48069
iteration 3: removed 904 (1.88%), remaining 47165
```

每轮删除数也会记录在 `pc_filter.removal_history`（list[int]），方便你审计或画收敛曲线。

## 多台风机

`PowerCurveFiltering` 一次只处理一台；多台由调用方 `groupby` 后循环调用：

```python
from wtclean import PowerCurveFiltering, estimate_machine_parameters

results = {}
for name, group in df.groupby("Wind_turbine_name"):
    params = estimate_machine_parameters(group, "Ws_avg", "P_avg")  # 先反推厂商参数
    pcf = PowerCurveFiltering(
        "Ws_avg", "P_avg", group,
        cut_in_speed=params["cut_in_speed"],
        rated_wind_speed=params["rated_wind_speed"],
        rated_power=params["rated_power"],
    )
    results[name] = pcf.process()   # (normal_df, abnormal_df)
```

## 厂商参数不知道怎么办

`cut_in_speed` / `rated_wind_speed` / `rated_power` 是**风机机型物理参数，必填**。若拿不到厂商技术参数，可先从 SCADA 数据反推（`estimate_machine_parameters` 会估计全部三个值），再把结果传给构造函数：

```python
from wtclean import estimate_machine_parameters

params = estimate_machine_parameters(turbine, "Ws_avg", "P_avg")
# -> {"cut_in_speed": 3.78, "rated_wind_speed": 13.59, "rated_power": 1985.3}
```

> 注意：反推值来自（通常是 10 分钟统计的）SCADA 数据，是"软"值——例如反推的额定风速约 13 m/s，会低于厂商标称的约 14.5 m/s，因为时间平均会把功率曲线拐点往左拉。生产使用前建议与厂商功率曲线表交叉核对；正式项目优先填厂商标称值。

## 参数说明

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `windspeed_label` | — | 风速列名（必填）。 |
| `power_label` | — | 有功功率列名（必填）。 |
| `df` | — | **单台**风机的 SCADA DataFrame（必填）。 |
| `cut_in_speed` | — | 切入风速 m/s（厂商参数，必填）。低于该风速默认保留；停机判定从该风速起算。 |
| `rated_wind_speed` | — | 额定风速 m/s（厂商参数，必填）。风速 ≥ 它时风机应接近额定功率，是"高风低功率粗洗"的起点。 |
| `rated_power` | — | 额定功率 kW（厂商参数，必填）。高风区各类比例阈值都以此为基准。 |
| `low_power_ratio` | `0.9` | 高风低功率粗洗阈值：风速 ≥ 额定风速时，功率 < 该比例×额定功率 即剔除。 |
| `bin_interval` | `0.5` | 风速分箱宽度 m/s。越小分箱越细（样本少的区段统计越不稳），越大越粗。 |
| `z_coeff` | `2.5` | 正常带宽度 = `中位数 ± z_coeff × MAD`。见下方调参建议。 |
| `filter_cycle` | `3` | bin+MAD 迭代精洗的轮数上限。见下方调参建议。 |
| `return_fig` | `False` | 是否保存一张功率曲线清洗结果散点图（蓝=Normal / 橙=Abnormal）。 |
| `image_path` | `""` | `return_fig=True` 时输出图片的**完整文件路径**（含文件名，如 `./images/turbine_pc.png`）；目录不存在会自动创建。 |

## 调参建议（最终决策权在你）

**"保留多少 / 洗得多纯"没有绝对正确的值**，取决于你拿到 `normal_df` 后要干什么——拿去给高阶模型精洗可以放宽一点（尽量保点），指望初筛结果直接用就得收紧。每次调参都是"保点 vs 纯净"的权衡，建议结合输出图与逐轮打印来判断。以下给出方向和量级参考。

### `z_coeff`——正常带多宽（影响最大，决定"删多少"）

含义：功率偏离该箱中位数多少个 MAD 算正常。**MAD 用原始值、未乘 1.4826**，所以不能直接当"σ 倍数"读；换算成高斯直觉（原始 MAD ≈ 0.675σ）：

| z_coeff | 正常带约 | 高斯下保留比例 | 档位 |
| --- | --- | --- | --- |
| `2.0` | ±1.35σ | ~82% | 激进（洗得纯，易误删） |
| `2.5`（默认） | ±1.69σ | ~91% | 居中 |
| `3.0` | ±2.02σ | ~96% | 偏宽松 |
| `4.0` | ±2.70σ | ~99% | 很宽松（基本只剔粗洗） |

- **越大 → 带越宽 → 保留越多**（正常点误删少，但漏进 `normal_df` 的真异常变多）；
- **越小 → 带越窄 → 删得越多**（`normal_df` 更纯，但可能误删边界正常点）。

实测参考（La Haute Borne，MM82，`filter_cycle=3`）：`z_coeff=2.5` 时 abnormal 约 13–15%；`z_coeff=4.0` 时骤降到约 3%（此时 MAD 层每轮只删 ~1%，第 2、3 轮基本空转）。也就是说 `z_coeff=4` 已接近"纯粗洗"。

建议：后续要做 GAM/sigmoid/时间维度精洗时，可把 `z_coeff` 放到 3~4 先把明显离群点去掉；若想让这一层初筛就尽量干净、愿意接受少部分正常点损失，用 2~2.5。

### `filter_cycle`——迭代几轮

机制：第 1 轮删得最多（清掉最明显的离群点），之后每轮重新分箱、带变窄，**删除量快速递减**——后面几轮主要是在不断收窄正常带，**可能开始误删边界正常点**。

- 实测各数据集上，多数风机的删除量到第 3~5 轮已降到每轮 <1%（可看逐轮打印确认）。
- 想要**更保守、尽量保留正常点**：`2` 就够，甚至 `1`（只粗洗 + 单轮 MAD）；
- 想**更彻底**（如训练样本允许损失一部分）：`5`，但要警惕过度清洗；
- 反正每轮都会打印删除量与比例，看到某轮删除已经趋近 0，就说明再加轮次意义不大。

### 其它参数

- **`low_power_ratio`**：只影响额定风速以上的"限电/降载"粗洗。想多剔除降载工况就调小（如 0.85），想更保险保留就调大。注意别设太低，否则额定平台上的轻微降载会漏掉。
- **`bin_interval`**：高风速区样本稀的话可适当调大（如 1.0）提高该区段 MAD 稳定性；低风速区样本极多时调小能更精细。默认 0.5 对多数 10 分钟数据够用。
- **`cut_in_speed`/`rated_wind_speed`/`rated_power`**：这三个是物理参数，**不要拿来当清洗旋钮调**。设错会直接让粗洗规则失效（比如额定风速设太低会把正常爬坡段误当"高风低功率"整段删掉）。拿不准就反推，再和厂商表核对。

## 出图（`return_fig=True`）

会画一张风速-功率散点图：蓝色 = Normal（`normal_df`），橙色 = Abnormal（`abnormal_df`），图例已标注。图片保存到 `image_path` 指定的完整文件路径。建议每次调参都出一张图，肉眼确认"边界处"是否删得合理。

## 测试

```bash
python -m unittest discover -s test -v
```

当前测试覆盖：正常/异常切分、重复索引不膨胀、粗洗（停机、高风低功率）、非法参数与空数据校验、出图、迭代删除记录。
