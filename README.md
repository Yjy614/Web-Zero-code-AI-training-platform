# 工业场景零代码 AI 训练平台 · Web 端

与桌面端（PyQt6）并行的 Web 产品。一期聚焦 **目标检测**（实例分割仅预留目录与资源 Tab）；默认开启 **演示模式**（Mock 训练，不自动下载权重）。关闭演示模式后可走本机 Ultralytics 真实训练。

## 目录

- `frontend/` — Vue 3 + TypeScript + Element Plus（向导、标注、资源页）
- `backend/` — FastAPI + SQLite（任务队列、数据集、模型库、权重）
- `backend/config/` — `web_config.yaml` 等（运行时设置见下方「配置」）
- `deploy/` — 部署示例
- `Web端项目说明书.md` — 产品与编码规格

本地运行会产生、且默认不入库的内容（见根目录 `.gitignore`）：

- `backend/storage/` — 数据集、训练 runs、导出、报告、模型归档
- `backend/pretrained/` — 管理员上传的预训练 `.pt`
- `backend/platform_web.db` — SQLite 库
- `frontend/node_modules/`、`frontend/dist/`

## 环境要求

- Node.js 20+（推荐 22 LTS）
- Python 3.10+
- 真实训练另需 `ultralytics`（及相应 torch，体积较大）

## 启动后端

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档：http://127.0.0.1:8000/docs

默认账号：

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | 管理员 |
| demo | demo123 | 使用人员 |

## 启动前端

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开：http://127.0.0.1:5173  
开发服务器已将 `/api` 代理到后端 `8000` 端口。

> 若 PowerShell 提示找不到 `npm`，先确认已安装 Node，并把 `D:\nodejs`（或你的安装路径）加入 PATH；若提示禁止运行脚本，可执行：  
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 主要能力（一期 · 检测）

- **7 步向导**：导入 → 清洗 → 标注 → 配置 → 训练 → 评估 → 导出
- **标注**：YOLO 矩形框；切换图片自动保存；AI 预标注（短训后补全未标注图）；训练成功后预标注视为用户标注
- **训练**：演示模式 Mock；关闭后本机真实训练；曲线与进度轮询
- **模型库**：按次归档（同名任务自动 `名称_2`、`名称_3`…）；下载 PT / 转 ONNX；按用户与任务类型隔离
- **资源与权限**：数据集 / 模型 / 权重仓库；管理员可上传预训练权重、管用户；角色隔离

## 当前里程碑

- **M1**：登录 / 侧栏 / 向导壳 / 演示横幅
- **M2**：数据集上传、清洗、YOLO 标注（`storage/datasets/detect/<用户名>/<名>/`）
- **M3**：配置 + Mock 训练 / 评估 / 导出
- **M4**：用户管理、设置页（LLM/Vision）、权限隔离
- **本机真实训练**：关闭演示模式后走 `LocalJobRunner`（Ultralytics）
- **增强**：AI 预标注、模型库 PT/ONNX、训练产物唯一命名与归档路径修正

## 真实训练（本机）

1. 安装依赖（体积较大，会拉取 torch）：

```powershell
cd backend
python -m pip install -r requirements.txt
```

2. 管理员在「权重仓库」上传真实 `.pt`（如 `yolo11n.pt`）到服务器 `pretrained/<任务类型>/`，**不会自动从公网下载**。
3. 设置中关闭「演示模式」。
4. 走完标注 → 配置（选已上传权重）→ 开始训练。

CPU 可跑但较慢；有 NVIDIA GPU 时设备选 `cuda:0`。

## 配置

见 `backend/config/web_config.yaml`。

- `storage_root`：用户数据（datasets / runs / exports / reports / **models**）
- `pretrained_root`：系统预训练权重（仅管理员上传，与 storage 分离；训练起点，不是模型库归档）

运行时设置（演示开关、大模型配置）会写入 `backend/config/runtime_settings.json`（已加入 `.gitignore`）。

额外依赖：`Pillow`（图片清洗）、`ultralytics`（真实训练 / 预标注）、`echarts`（训练曲线，前端依赖）。

## 说明

- 一期 **不做** 实例分割完整链路（可留占位）；不做浏览器自动下权重、不做公网 SaaS。
- Clone 后需自行安装依赖、启动服务；权重与数据集需本机准备或管理员上传。
