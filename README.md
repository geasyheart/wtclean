# wtclean

风电机组 SCADA 功率曲线数据清洗工具：用「迭代分箱 + MAD（中位数绝对偏差）」把**正常运行数据**从原始 SCADA 数据中初筛出来。

## 这个包是做什么的

输入**一台风机**的 SCADA 数据（至少要含风速、有功功率两列，通常为 10 分钟统计记录）。

`PowerCurveFiltering.process()` 会把数据按行切分成两份返回：

- **`normal_df`**：风机**正常运行**的数据点——功率落在该风速区间应有的功率带内。**后续做功率曲线建模、工况分析、机器学习等时主要用这一份**。
- **`abnormal_df`**：其余所有被剔除的点。⚠️ **它不是"严格异常集"**，而是混了多种情况：停机、故障、限电/降载、桨距控制等**其它工况**、以及真正的传感器/数据异常。想区分异常类型，需要后续用更高阶方法（如 GAM/sigmoid 拟合、时间维度分析）处理。

> 想识别**限电/降载/停机成片出现、单点看却落在正常带内**的"隐藏异常"，并把异常天整体剔除、输出 `normal/abnormal` 两个子集，请用 [DailyAnomalyDetector](#dailyanomalydetector时间维度异常检测v2)（v2，时间维度 + 基准曲线对比）。`PowerCurveFiltering`（v1）是无模型的 bin+MAD 粗洗，适合作为对照或初筛。

单台风机的整条清洗流水线分三步：

1. **去停机**：风速 ≥ 切入风速、但功率 ≤ 1 kW 的点（风机已停机）→ 剔除。
2. **高风低功率粗洗**：风速 ≥ 额定风速、但功率 < `low_power_ratio × 额定功率` 的点（限电、降载、桨距/故障等非正常发电）→ 剔除。
3. **迭代 bin+MAD 精洗**：按 `bin_interval` 把风速分箱，对每箱算功率中位数与 MAD，箱内功率落在 `中位数 ± z_coeff × MAD` 带外的点 → 剔除。剔除后**重新分箱再剔**，重复 `filter_cycle` 轮（每轮带会随数据变干净而收窄）。

几个内置的判定约定（避免误杀正常点）：

- 箱内样本 < 3，或功率离散度为 0（典型如额定功率平台），该箱**不判定、直接保留**；
- 风速低于切入风速的点默认保留（近零功率属正常待机）；
- 恒定功率的"水平聚集"类异常（如某段时间被固定在某一功率）**bin+MAD 识别不出来**，这是本方法的能力上限，应交给后续更精细的方案。

### 快速开始

```python
import pandas as pd
from wtclean import DailyAnomalyDetector

df = pd.read_csv("data.csv")
turbine = df[df["Wind_turbine_name"] == "R80721"]   # 单台风机

detector = DailyAnomalyDetector(
    windspeed_label="Ws_avg",     # 风速列名
    power_label="P_avg",          # 有功功率列名
    timestamp_label="Date_time",  # 时间戳列名（datetime 或可解析字符串）
    df=turbine,                   # 单台风机数据
    # —— 以下物理参数缺省时会从 SCADA 自动反推（见下）——
    cut_in_speed=3.5,             # 切入风速 m/s
    rated_wind_speed=14.5,        # 额定风速 m/s
    rated_power=2050,             # 额定功率 kW
    cut_out_speed=25.0,           # 切出风速 m/s（可选；给定后切出以上近零功率视为正常停机）
)

normal_df, abnormal_df = detector.detect()

detector.abnormal_days_    # set：被判异常的天（日期）
detector.normal_days_      # set：被判正常的天（日期）
```

### 输出含义

`detect()` 返回 `(normal_df, abnormal_df)` 两个 DataFrame，按行切分原始 `df`（`normal_df + abnormal_df == 原始 df`，且索引不交叠）：

- **`normal_df`**：正常天内的正常点（粗洗异常点不会进入）。
- **`abnormal_df`**：其余所有点，包括粗洗异常点、被天级判定标出的异常天、以及样本数异常的天。

天级判定只用一个固定规则：`median ± 3×MAD`（稳健 3σ），不再暴露额外的逐点残差参数。

### 检测流程

1. **粗洗掩码**：`nan` / `range` / `downtime` / `curtailment` 直接判异常，不参与后续建模。
2. **整机基准曲线**：用 `base_method`（默认 `gam` 单调 GAM，也可 `logistic_5_parametric` / `iec`）在粗洗后的点上拟合。
3. **残差**：`resid = power - base(wind_speed)`。
4. **天级判定**：按天聚合中位残差和样本数；先用 `median ± 3×MAD` 过滤样本数异常的天，再对剩余天的中位残差用 `median ± 3×MAD` 判正常天/异常天。
5. 输出 `(normal_df, abnormal_df)`。

### 参数说明

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `windspeed_label` / `power_label` / `timestamp_label` | — | 风速/功率/时间戳列名（必填）。 |
| `df` | — | **单台**风机的 SCADA DataFrame（必填）。 |
| `cut_in_speed` / `rated_wind_speed` / `rated_power` | 自动反推 | 厂商物理参数；缺省时调用 `estimate_machine_parameters` 反推。 |
| `cut_out_speed` | `None` | 切出风速 m/s；给定后切出风速以上功率归零视为正常停机，未给则把高风零功率当停机异常。 |
| `low_power_ratio` | `0.9` | 限电粗洗阈值：风速 ≥ 额定风速时功率 < 该比例×额定功率即判限电。 |
| `downtime_power` | `1.0` | 功率 ≤ 该值（kW）且风速在切入以上视为停机。 |
| `base_method` | `"gam"` | 基准曲线后端：`gam`（默认，单调 GAM）/ `logistic_5_parametric` / `iec`。 |
| `**base_kwargs` | — | 透传给 `fit_power_curve` 的额外参数。 |

### 调参建议

- **天级阈值固定为 `median ± 3×MAD`**，不再暴露额外参数；需要更松/更紧时直接改 `detect()` 里的 `k = 3.0`。
- **`base_method`**：默认 `gam` 带单调约束 + 网格搜样条数，无需调；`logistic_5_parametric` 的默认 bounds 假设 1.2–1.8 MW，其它机型需显式传 bounds。
- 每天样本数也按 `median ± 3×MAD` 过滤；若某些天只有个别点，这些天会整体进入 `abnormal_df`。这符合"下游只用 `normal_df`、宁可多删"的默认取舍。

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

### 反推值 vs 厂商值的偏差有多大（La Haute Borne 实测）

La Haute Borne 四台 Senvion MM82，厂商参数 `3.5 / 14.5 / 2050`（切入/额定风速/额定功率），`estimate_machine_parameters` 反推结果与影响如下：

| 风机 | 反推 cut_in | 反推额定风速 | 反推额定功率 | 用反推 vs 厂商的判定差异 |
| --- | --- | --- | --- | --- |
| R80711 | 3.79 | 12.54 | 1930 | 0.40% 行 |
| R80721 | 3.79 | 12.72 | 1985 | 0.51% 行 |
| R80736 | 3.78 | 12.71 | 2005 | 0.61% 行 |
| R80790 | 3.67 | 13.01 | 1989 | 0.40% 行 |

两点值得注意：

1. **额定风速反推值（约 12.5–13.0）确实比厂商标称（14.5）低约 1.5–2 m/s**，这是 10 分钟平均口径的固有"软"偏差。但在这批数据上它**几乎不影响最终结果**：因为接近额定风速后真实功率已基本贴满 0.9×额定功率，粗洗规则在高风速段的触发对额定风速不敏感。
2. 反推与厂商两种做法**判定的差异行只有约 0.4–0.6%**，且几乎全部集中在**切入风速附近的功率≈0 边界点**（差异行中位风速 ≈3.6 m/s、中位功率 = 0）——本质是反推切入风速偏高一点点（3.7~3.8 vs 3.5）导致"停机"判定边界挪了几十厘米风速。**对 `normal_df` 的功率曲线主体毫无影响。**

> 结论：反推参数在这类"正常运行占比高"的数据上完全可用，对结果影响很小。但**若数据里限电/降载占比高**，额定风速偏低的反推值会让"高风低功率粗洗"提前到功率爬坡段触发，可能多删一批合法的低功率点——这种情况下务必用厂商标称参数，或把反推额定风速往上修正。

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

实测参考（La Haute Borne，MM82）：`filter_cycle=3` 时，`z_coeff=2.5` → abnormal ≈ 13%；`z_coeff=4.0` → ≈ 3%（此时 MAD 层每轮只删 ~1%，第 2、3 轮基本空转，接近"纯粗洗"）。`z_coeff` 与 `filter_cycle` 是**强耦合**的，怎么配看下面的网格实测。

### `filter_cycle` 与 `z_coeff` 的配合（La Haute Borne R80721 实测）

下表为"abnormal 占比"，横轴 `z_coeff`、纵轴 `filter_cycle`：

| cycle＼z | `2.0` | `2.5` | `3.0` | `4.0` |
| --- | --- | --- | --- | --- |
| `1` | 15.0% | 8.7% | 5.1% | 2.6% |
| `2` | 22.6% | 11.7% | 6.1% | 2.7% |
| `3` | 27.7% | 13.4% | 6.5% | 2.7% |
| `5` | 34.3% | 14.7% | 6.7% | 2.7% |
| `8` | 40.7% | 15.0% | 6.7% | 2.7% |

从这张表能直接读出三条规律（这也是"该不该加迭代轮数"的判据）：

1. **带越宽，收敛越快**：`z=4` 第 2 轮后删除就降到 0.14%→0.02%→0，第 2 轮即可停；`z=3` 约第 3–4 轮收敛；`z=2.5` 到第 5 轮才基本收敛（继续到第 8 轮 abnormal 只从 13.4% 涨到 15.0%，也就是多删了约 900 个正常/边界点）。
2. **`z=2` 是危险区**：每轮删除量不衰减（13.7%→8.9%→6.6%→5.1%→4.2%……），跑到 8 轮时 abnormal 已到 40%。这说明带太窄、装不下数据真实的离散度，**加轮数只会不断"吃"正常点**。遇到这种逐轮删除不衰减的情况，别再加大 `filter_cycle`，应调大 `z_coeff`（或换 `doc/` 下的 GAM/sigmoid 方法）。
3. **看逐轮打印做决定**：每轮都打印 `removed X (Y%), remaining Z`。某轮删除已 < 0.5% → 再加轮意义不大；连跑好几轮删除都 > 1% 不衰减 → 带太窄，往回调 `z_coeff`。

### 不同情况怎么配（结合上面实测的建议档）

- **只想粗洗、后续交给高阶方法精洗**（`normal_df` 允许混少量离散点）：`z_coeff=3~4`、`filter_cycle=2~3`。La Haute Borne 下 abnormal 约 3–7%，基本只剔停机/高风低功率/明显离群点，正常点保留最全。
- **想让 bin+MAD 这一层尽量洗纯**（`normal_df` 直接用于建模，能接受少量正常点损失）：`z_coeff=2.5`、`filter_cycle=5`。abnormal 约 15%，注意第 4–5 轮起每轮删 <1%，若不想误删边界点可提前手动用小轮次。
- **数据很干净 / 风机台数多样本足**：可收紧 `z_coeff=2.0~2.5`；反之**样本少、高风速 bin 稀疏**时统计不稳，应放宽 `z_coeff≥3` 并加大 `bin_interval`。
- **数据污染重 / 工况杂**（大量限电、降载、停机混在正常里，如部分比赛风机）：bin+MAD 会把"偏离主体"的正常工况也算成异常，**别指望靠调小 `z_coeff` 或加 `filter_cycle` 洗得更准**——这类数据优先保证 `normal_df` 不被误杀，用 `z_coeff=4` 只剔粗洗，真正的异常分类交给 `doc/` 下的按天拟合/模型方案。

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
