# Deep Interview Spec: MLflow Integration for PaddleOCR Training

## Metadata
- Interview ID: di-mlflow-paddleocr-001
- Rounds: 3
- Final Ambiguity Score: 14.5%
- Type: brownfield
- Generated: 2026-03-24
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 35% | 0.333 |
| Constraint Clarity | 0.70 | 25% | 0.175 |
| Success Criteria | 0.85 | 25% | 0.213 |
| Context Clarity | 0.90 | 15% | 0.135 |
| **Total Clarity** | | | **0.855** |
| **Ambiguity** | | | **14.5%** |

## Goal
为 PaddleOCR 的训练流程添加完整的 MLflow 实验跟踪功能，支持连接远程 MLflow Tracking Server，记录完整的实验记录（指标、模型、配置文件、训练日志等全部产物）。

## Constraints
- 使用远程 MLflow Tracking Server（非本地文件存储）
- 用户不关心 W&B，可以共存或忽略
- 需要遵循现有的 BaseLogger 抽象接口
- YAML 配置文件需要支持 MLflow 相关选项
- **待确认**: 认证方式（token/credentials）、服务器 URL 配置方式（环境变量 vs 配置文件 vs CLI）

## Non-Goals
- 不需要修改 W&B 相关代码
- 不需要实现 MLflow Model Registry 功能（除非用户后续要求）
- 不需要支持本地文件存储模式

## Acceptance Criteria
- [ ] 创建 `MLflowLogger` 类，继承 `BaseLogger` 抽象类
- [ ] 在 YAML 配置中添加 MLflow 相关选项（`logger_type: mlflow`, `mlflow_tracking_uri`, `mlflow_experiment_name` 等）
- [ ] 训练过程中记录指标：loss, learning rate, batch time, IPS, memory, eval metrics
- [ ] 训练结束时记录产物：模型权重 (.pdparams)、优化器状态 (.pdopt)、训练状态 (.states)、配置信息 (.info.json)
- [ ] 记录训练配置参数（超参数、数据集配置等）
- [ ] 训练日志文件作为 artifact 记录
- [ ] 可通过 `python tools/train.py -c config.yml --mlflow` 或配置文件启用 MLflow
- [ ] 在远程 MLflow Server 的 UI 中可看到完整的实验记录

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| 用户可能只想记录基本指标 | 询问"最低限度的成功是什么" | 用户明确需要完整实验记录 |
| 可能需要与 W&B 共存 | 询问与 W&B 的关系 | 用户不关心 W&B |
| 可能使用本地文件存储 | 询问后端存储方式 | 用户明确使用远程服务器 |

## Technical Context

### 现有代码架构
- **训练入口**: `tools/train.py` → `tools/program.py` 中的 `train()` 函数
- **Logger 抽象**: `ppocr/utils/loggers/` 目录
  - `BaseLogger`: 抽象基类，定义 `log_metrics()`, `log_artifact()`, `log_params()`, `close()` 等接口
  - `WandbLogger`: W&B 实现，可作为参考模式
- **配置系统**: YAML 文件驱动，支持 `-o key=value` 运行时覆盖
- **指标**: loss, lr, batch_time, IPS, memory, eval_metrics
- **Checkpoint 文件**:
  - `.pdparams`: 模型权重
  - `.pdopt`: 优化器状态
  - `.states`: 训练状态 (pickle)
  - `.info.json`: 元数据

### 实现路径
1. 创建 `ppocr/utils/loggers/mlflow_logger.py`
2. 在 `ppocr/utils/loggers/__init__.py` 中注册 MLflowLogger
3. 修改 `tools/program.py` 中的 logger 初始化逻辑
4. 添加 YAML 配置选项支持

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| MLflow | ExternalService | tracking_uri, experiment_name, run_id | implements LoggerInterface, connects via MLflowLogger |
| MLflowLogger | Class | client, run, experiment_id, tracking_uri | extends BaseLogger, uses MLflow |
| TrainingPipeline | Module | entry_point, logger_instance, config | uses BaseLogger, loads YAMLConfiguration, produces TrainingMetrics, saves Checkpoint |
| BaseLogger | AbstractClass | log_metrics(), log_artifact(), log_params(), close() | implemented_by MLflowLogger, implemented_by WandbLogger |
| WandbLogger | Class | project, entity, run | extends BaseLogger, can coexist or be replaced |
| YAMLConfiguration | ConfigurationFile | mlflow_tracking_uri, mlflow_experiment_name, logger_type | configures MLflowLogger, loaded_by TrainingPipeline |
| Checkpoint | Artifact | model_path, optimizer_path, states_path, info_json | logged_by MLflowLogger, produced_by TrainingPipeline |
| TrainingMetrics | DataStructure | loss, lr, batch_time, IPS, memory, eval_metrics | logged_by MLflowLogger, produced_by TrainingPipeline |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 6 | 6 | - | - | - |
| 2 | 8 | 2 | 0 | 6 | 75% |
| 3 | 8 | 0 | 0 | 8 | 100% |

## Interview Transcript
<details>
<summary>Full Q&A (3 rounds)</summary>

### Round 1
**Q:** 你希望 MLflow 与现有的 W&B (Weights & Biases) 日志系统是什么关系？
**A:** 不在乎 W&B，只想要基本的 MLflow 实验跟踪
**Ambiguity:** 54.5% (Goal: 0.5, Constraints: 0.3, Criteria: 0.4, Context: 0.7)

### Round 2
**Q:** MLflow 的后端存储方式是什么？你打算用本地文件还是远程 MLflow Server？
**A:** 远程服务器
**Ambiguity:** 41% (Goal: 0.7, Constraints: 0.5, Criteria: 0.4, Context: 0.8)

### Round 3
**Q:** 你希望 MLflow 记录哪些内容？最低限度的"成功"是什么样的？
**A:** 完整实验记录（指标、模型、配置文件、训练日志等全部产物）
**Ambiguity:** 14.5% (Goal: 0.95, Constraints: 0.7, Criteria: 0.85, Context: 0.9)

</details>
