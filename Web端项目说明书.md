# Web 端项目说明书（指导 AI 编码）

> 本文档用于指导 AI / 开发者实现「工业场景零代码 AI 训练平台」的 **Web 端**。  
> 与现有 **桌面客户端（PyQt6）并行**，不替代桌面端。  
> 编码时以本文为准；与桌面行为冲突时，以「当前实现状态」与「一期范围」章节为准。

---

## 当前实现状态（相对原文已演进）

以下能力**已落地**，编码时勿再按「仅占位」处理：

| 能力 | 说明 |
|------|------|
| 目标检测 | 7 步向导 + 矩形框标注 + Mock/本机训练 |
| 实例分割 | 7 步向导 + 多边形标注 + Mock/本机训练（与 detect 对齐） |
| 本机真实训练 | `LocalJobRunner` + Ultralytics；关闭演示模式后启用 |
| 产物版本隔离 | `runs/exports/reports/models/<tt>/<user>/<task>/<run_key>/` |
| 模型库 | PT/ONNX 下载；无 ONNX 可转格式；删除清理对应版本产物 |
| 推理试用 | 模型库选模型 → 上传图片 → 服务端推理并返回可视化图 |
| AI 预标注 | 短训临时权重后补全未标注图 |

仍不做：OCR、公网自动下权重、SaaS、Windows 推理 EXE、真实集群调度（仅预留接口）。

---

## 0. 给 AI 的硬约束（先读）

1. **任务类型**：已支持 `detect` 与 `segment`；OCR **不做 UI、不写业务接口**。
2. **与桌面检测/分割流程功能对齐**（7 步向导 + 资源管理 + 设置中与训练相关的部分）。
3. **演示模式**：默认开启——**禁止自动下载任何 `.pt` 权重**；训练/评估可走 **Mock 执行器** 跑通全流程页面。
4. **技术栈固定**：前端 **Vue 3 + TypeScript**；后端 **Python（FastAPI 推荐）**；尽量复用桌面 `core/` 中与检测相关的数据约定与算法思路。
5. **部署形态**：工厂 **内网私有化**；浏览器只做交互与下发；训练将来由后端调度 **显卡集群**（一期用 Mock/本地队列占位）。
6. **必须登录**；角色仅两类：**管理员（admin）**、**使用人员（user）**。
7. 注释与用户可见文案用 **中文 UTF-8**；代码标识符用英文。
8. 不要实现桌面端的「打包推理 EXE」到 Web；导出改为服务器上的文件下载（PT/ONNX）；在线试用走「推理试用」页。

---

## 1. 背景与目标

### 1.1 桌面端现状（对照基准）

仓库：`Zero-code-AI-training-platform`（本地离线 PyQt6 + Ultralytics）。

- 侧栏插件：目标检测训练、实例分割训练（OCR 关闭）。
- 检测 7 步：导入 → 清洗 → 标注 → 配置 → 训练 → 评估 → 导出。
- 全局：数据集管理、模型库、权重仓库、系统设置（含 LLM / 视觉模型配置）。
- 数据按 `detect` / `segment` 隔离；无登录体系。

### 1.2 Web 端目标

| 目标 | 说明 |
|------|------|
| 并行产品 | 与桌面端并存，独立仓库或 monorepo 的 `web/` 目录均可 |
| 当前能力 | **检测 + 实例分割** 向导与资源管理；推理试用；本机真实训练 |
| 中期架构 | 后端部署在内网服务器；数据集落在集群可挂载的共享存储；训练任务下发到 GPU 集群 |
| 非目标 | OCR、权重在线下载、公网 SaaS、本机 GPU 直连浏览器、打包推理 EXE |

### 1.3 成功标准（领导演示）

- 可登录，角色权限可见差异。
- 能走完检测 7 步 UI（含进度、图表、报告页），数据在服务器可持久化。
- **不依赖外网下载权重**；无真实 GPU 时 Mock 训练仍能演示。
- 文案与信息架构接近桌面端，降低讲解成本。

---

## 2. 一期范围（历史基线；实现已超越部分条目）

### 2.1 必须做（P0）

- 登录 / 登出 / 会话；admin、user 角色。
- 检测训练向导 7 步（Web UI + API）。
- 检测数据集上传到服务器、列表、删除（权限内）。
- 资源管理：数据集、模型库、预训练权重列表（可只读展示占位）。
- 系统设置：通用项 + 大模型/视觉模型配置（存服务器；演示期可不真实调用）。
- 演示模式开关（配置项）：Mock 训练/评估/导出。
- 任务队列抽象：`JobRunner` 接口 + `MockJobRunner`；预留 `ClusterJobRunner`。

### 2.2 明确不做

- OCR 插件。
- 浏览器端自动下载 / 拉取 Ultralytics 官方权重。
- 打包 Windows 推理 EXE。
- 多租户 SaaS、复杂审批流。
- 真实集群调度实现（只留接口与配置项）。
- MobileSAM 专用链路（分割走 YOLO-seg 即可）。

### 2.3 已实现 / 进行中（相对原文「后期预留」）

- `segment` 任务类型、分割向导、多边形标注、分割训练与导出。
- 真实权重管理（管理员上传 `.pt` 到 `pretrained/<tt>/`，禁止公网下载）。
- 本机 `LocalJobRunner` 真实训练。
- 训练产物版本隔离与模型库管理。
- **推理试用页**（选模型库模型 + 上传图 → 可视化结果）。

### 2.4 后期预留

- `ClusterJobRunner`：对接工厂 GPU 集群（Slurm / K8s / 自研调度均可）。

---

## 3. 角色与权限

| 角色 | 能力 |
|------|------|
| **admin** | 用户管理（增删改、重置密码）；全局数据集/模型查看与删除；系统设置；演示模式开关；权重占位管理 |
| **user** | 登录后使用检测向导；管理自己创建的数据集与训练任务；查看自己的模型/导出；不可改系统级配置（或只读） |

规则建议：

- 所有业务 API 需鉴权（JWT 或 Session Cookie，内网 HTTPS 可选）。
- 数据集/训练任务带 `owner_id`；user 只能操作自己的；admin 可操作全部。
- 默认内置账号（安装时种子数据）：`admin / 初始密码可配置`；至少一个演示 `user`。

---

## 4. 总体架构

```
┌─────────────┐     HTTPS/内网      ┌──────────────────────────────┐
│  Vue3 浏览器 │ ◄────────────────► │  Python API 服务（FastAPI）   │
│  交互 / 下发  │                    │  鉴权 · 业务 · 文件 · 队列    │
└─────────────┘                    └──────────────┬───────────────┘
                                                  │
                       ┌──────────────────────────┼──────────────────────────┐
                       ▼                          ▼                          ▼
              ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────────┐
              │ 共享存储（推荐）   │      │ 元数据库          │      │ JobRunner            │
              │ datasets/runs/   │      │ SQLite/PostgreSQL│      │ Mock（一期）          │
              │ exports/models   │      │ users/jobs/...   │      │ Cluster（后期）       │
              └─────────────────┘      └─────────────────┘      └─────────────────────┘
                                                  │ 后期
                                                  ▼
                                       ┌─────────────────────┐
                                       │ 工厂 GPU 集群        │
                                       │ 读取共享盘数据集训练  │
                                       └─────────────────────┘
```

### 4.1 为何数据集要上服务器

- 浏览器无法直接把本地盘挂到集群。
- 数据集上传到 **API 服务器可访问、且将来集群也能挂载的共享目录**，便于重复训练与多人协作。
- 与桌面 `data/datasets/detect/<名>/` 目录结构 **尽量一致**，便于复用逻辑与以后互通。

### 4.2 后端部署关系（回答架构判断）

**是的**：Web 后端部署在内网服务器上；浏览器只调 API。  
训练时由后端创建 Job，**将来**把 Job 交给集群调度器，在挂载同一份数据的计算节点上跑 Ultralytics。  
一期用 Mock：不调 GPU，定时推送假进度与假指标，写一份假的 `best.pt` 占位文件或仅写库记录。

### 4.3 与桌面代码复用建议

| 可复用思路 | 说明 |
|------------|------|
| 目录约定 | `datasets/detect/...`、`runs/detect/...`、`exports/detect/...` |
| 标注格式 | YOLO txt 检测框 |
| 清洗算法 | 长边归一 + 感知哈希去重（可移植为服务端函数） |
| 划分与 yaml | `prepare_yolo_yaml` 同类逻辑 |
| 不宜直接复用 | PyQt UI、`pack_inference_exe`、桌面无用户的 DB 表结构需扩展 |

推荐：新建 `web/` 或独立仓库 `xxx-web`，后端 `app/` 内 **移植/包装** 检测相关 core，而不是强行 import 整桌面 UI。

---

## 5. 功能规格（对齐桌面检测）

### 5.1 信息架构（前端路由建议）

```
/login
/app
  /detect/wizard          # 检测 7 步向导（可带 taskId）
  /resources/datasets
  /resources/models
  /resources/weights
  /settings               # admin 可写；user 只读或隐藏敏感项
  /admin/users            # 仅 admin
```

侧栏文案对齐桌面：「目标检测训练」「数据集管理」「模型库」「基础模型权重仓库」「系统设置」。  
**不要**出现「实例分割训练」入口（或灰显「即将推出」且不可点）。

### 5.2 向导七步（P0 行为）

共用服务端「训练任务」对象 `TrainTask`（对应桌面 `wizard.task_data`）。

| 步骤 | 页面要点 | 服务端行为 |
|------|----------|------------|
| 1 导入 | 任务名；新建/选择数据集；上传图片（多文件或 zip） | 创建 `datasets/detect/<名>/images`；写 DB；关联 task |
| 2 清洗 | 一键清洗；展示保留/移除；可恢复 | 服务端执行清洗；结果落盘 |
| 3 标注 | 矩形框画布；类别 CRUD；翻页；保存 | YOLO labels；**一期保留手动框标注**（无 SAM）。VLM 预标注：可做按钮，演示模式可 Mock 或调用已配置的内网 VLM |
| 4 配置 | 划分比例、预训练权重选择、epochs、增强、设备、batch、imgsz | 生成 `data.yaml`；记录超参。权重列表可来自管理员上传清单或内置占位名（**不下载**） |
| 5 训练 | 开始/停止；进度；loss/mAP 曲线 | 创建 Job → JobRunner；WebSocket 或轮询推送进度 |
| 6 评估 | 指标卡、样例预览、报告、AI 建议 | Mock 或真实 eval；报告 HTML/JSON 可下载 |
| 7 导出 | PT/ONNX；下载链接 | 演示模式生成占位文件或跳过真实转换并提示「演示文件」 |

步骤校验：与桌面类似——未完成前置步骤不可进入下一步（可配置 `debug.free_step_nav` 便于开发）。

### 5.3 资源管理

- **数据集**：列表、详情（图片数/类别）、删除（权限）、进入向导。
- **模型库**：扫描或索引 `runs/detect/**` 与 DB 中的模型记录。
- **权重仓库**：列出 `models/detect/` 下文件；无文件时显示「未配置，请管理员上传」——**不要触发下载**。

### 5.4 设置

- 本地环境信息：API 版本、存储路径、演示模式状态（只读展示为主）。
- LLM / Vision：base_url、api_key、model、timeout（存服务器加密或权限保护）。
- 演示模式：`demo_mode: true`（默认 true）。

---

## 6. 演示模式 vs 真实模式

| 能力 | `demo_mode=true`（默认） | `demo_mode=false`（后期） |
|------|--------------------------|---------------------------|
| 权重 | 仅本地/共享盘已有文件；缺失则提示管理员上传 | 同左（始终禁止公网自动下载） |
| 训练 | `MockJobRunner`：30–60s 假进度 + 假曲线 | `ClusterJobRunner` 或本机 Ultralytics |
| 评估/导出 | Mock 指标与占位产物 | 真实评估与导出 |
| VLM 预标注 | 可选 Mock 框 / 或调用内网 VLM | 真实调用 |

配置示例（`config/web_config.yaml`）：

```yaml
demo_mode: true
storage_root: "/data/ai-platform"   # 集群可挂载的根路径
job_runner: "mock"                  # mock | local | cluster
forbid_weight_download: true        # 恒为 true
```

---

## 7. 数据模型（最小集）

### 7.1 存储目录（相对 `storage_root`）

```
storage_root/
  datasets/detect/<dataset_name>/
    images/
    labels/
    meta.json
    data.yaml
    removed/                 # 清洗移除
  runs/detect/<task_name>/
    weights/best.pt          # 真实或占位
    ...
  exports/detect/<task_name>/
  models/detect/             # 预训练权重（管理员上传）
  reports/detect/<task_name>/
```

### 7.2 数据库表（建议）

- `users`：id, username, password_hash, role(`admin`|`user`), created_at
- `datasets`：id, name, path, task_type=`detect`, owner_id, classes_json, created_at, updated_at
- `train_tasks`：id, name, owner_id, dataset_id, status, step, config_json, model_path, created_at
- `jobs`：id, task_id, type(`train`|`eval`|`export`|`clean`|`preannotate`), status, progress, message, result_json
- `models`：id, name, path, task_type, owner_id, metrics_json, created_at
- （可选）`settings`：key-value 或 JSON 文档

桌面现有 `platform.db` 无用户表，Web **不要直接共用同一文件** 除非做迁移方案；建议 Web 独立 DB。

---

## 8. API 大纲（AI 实现时按此拆路由）

统一前缀：`/api/v1`。除 `/auth/login` 外均需登录。

### 8.1 认证

- `POST /auth/login` → token
- `POST /auth/logout`
- `GET /auth/me`

### 8.2 用户（admin）

- `GET/POST /users`
- `PATCH/DELETE /users/{id}`

### 8.3 数据集

- `GET /datasets?task_type=detect`
- `POST /datasets`（创建元数据）
- `POST /datasets/{id}/images`（multipart 上传）
- `POST /datasets/{id}/images/zip`
- `POST /datasets/{id}/clean`
- `POST /datasets/{id}/restore`
- `GET/PUT /datasets/{id}/classes`
- `GET/PUT /datasets/{id}/annotations/{image}`
- `DELETE /datasets/{id}`

### 8.4 训练任务 / 向导

- `POST /tasks` / `GET /tasks` / `GET /tasks/{id}`
- `PATCH /tasks/{id}`（更新当前步配置）
- `POST /tasks/{id}/split`（划分并写 yaml）
- `POST /tasks/{id}/train` → job_id
- `POST /tasks/{id}/eval` → job_id
- `POST /tasks/{id}/export` → job_id
- `GET /jobs/{id}`；`GET /jobs/{id}/events`（SSE/WebSocket）

### 8.5 资源与设置

- `GET /weights?task_type=detect`
- `POST /weights/upload`（admin）
- `GET /models`
- `GET/PUT /settings`
- `GET /system/info`

错误响应统一：`{ "code": "...", "message": "中文说明" }`。

---

## 9. 前端页面规格（便于生成 Vue）

### 9.1 技术建议

- Vue 3 + Vue Router + Pinia + Axios
- UI：Element Plus 或 Naive UI（内网风、清晰步骤条）
- 标注页：Canvas 2D 或 Konva；支持画框、选中、Delete 删除、滚轮缩放（对齐桌面检测标注体验）
- 训练曲线：ECharts
- 鉴权：路由守卫；admin 路由单独 meta

### 9.2 关键交互

- 向导顶部 **步骤条**（与桌面 step_bar 文案一致）。
- 大文件上传：分片或至少进度条；失败可重试。
- 训练中禁止关闭任务或给出明确提示；支持取消（Mock 可立即停）。

---

## 10. 推荐仓库目录结构

```
ai-platform-web/                 # 或桌面仓库下 web/
├── README.md
├── docs/
│   └── Web端项目说明书.md       # 本文
├── frontend/                    # Vue3
│   ├── src/
│   │   ├── views/
│   │   ├── components/
│   │   ├── api/
│   │   ├── stores/
│   │   └── router/
│   └── package.json
├── backend/                     # FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/                # 配置、安全、存储路径
│   │   ├── models/              # ORM
│   │   ├── services/            # dataset/train/job...
│   │   ├── runners/             # mock.py, local.py, cluster.py(占位)
│   │   └── schemas/
│   ├── requirements.txt
│   └── tests/
└── deploy/
    ├── docker-compose.yml       # 可选：api + 静态资源
    └── env.example
```

---

## 11. 里程碑

| 阶段 | 交付 | 验收 |
|------|------|------|
| M1 壳子 | 登录、布局、侧栏、空向导页、演示横幅 | 领导可点开完整导航 |
| M2 数据 | 上传/列表/清洗/标注保存 | 服务器上可见 YOLO 目录结构 |
| M3 训练演示 | 配置页 + Mock 训练/评估/导出全流程 | 无 GPU、无外网跑通演示 |
| M4 权限与设置 | 用户管理、设置页、权限隔离 | admin/user 行为符合第 3 章 |
| 本机训练 | LocalJobRunner + 权重仓库上传 | 关闭演示模式可出真实 `best.pt` |
| 实例分割 | 分割向导 + 多边形标注 + 训练导出 | 与检测同等可演示 |
| 产物隔离 | runs/exports/reports/models 按 run_key 分目录 | 同任务多次训练互不覆盖 |
| 推理试用 | 模型库选模型 + 上传图推理 | 返回可视化图与检测列表 |
| M5（后期） | ClusterJobRunner 集群调度 | 集群出 `best.pt` |

---

## 12. 编码实施顺序（给 AI 的默认任务拆分）

1. 初始化 frontend/backend 工程与本文目录。  
2. 实现认证 + 用户种子数据。  
3. 实现存储路径层与 datasets CRUD + 上传。  
4. 实现清洗与标注 API + 标注 Vue 页。  
5. 实现 TrainTask 状态机与步骤 API。  
6. 实现 MockJobRunner + 进度推送 + 训练/评估/导出页。  
7. 资源管理三页 + 设置 + 演示模式横幅。  
8. 补齐权限测试与 README（部署到内网的最小说明）。

每完成一块：更新 API 与前端路由，保持可演示。

---

## 13. 与桌面端差异速查

| 项 | 桌面 | Web 当前 |
|----|------|----------|
| 登录 | 无 | 必须 |
| 任务类型 | detect + segment | detect + segment |
| 训练位置 | 本机 GPU | Mock / 本机 LocalJobRunner → 将来集群 |
| 数据位置 | 本机 `data/` | 服务器共享盘 |
| 权重 | 本地 `models/`，可能被 Ultralytics 下载 | **禁止下载**；管理员上传或占位 |
| 导出 EXE | 有 | 无；仅文件下载 + 推理试用页 |
| SAM | 分割有 | 无（YOLO-seg） |

---

## 14. 开放决策（已拍板）

- 与桌面 **并行**；一期 **只做检测**；分割/OCR **后期**。  
- **工厂内网私有化**；数据集 **上传到服务器（集群可挂载）**。  
- 角色：**管理员 / 使用人员**。  
- 前端 **Vue**，后端 **Python**。  
- 当前以 **演示壳** 为主，架构预留集群训练。

---

## 15. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-08-14 | 初版：基于桌面检测能力与产品确认纪要 |
