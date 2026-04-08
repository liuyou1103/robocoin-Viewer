# 🤖 RoboCoin Viewer - 数据清洗与元数据标注系统

## 📌 项目简介
专为**具身智能与机器人操作数据集**设计的智能处理系统，解决数据格式混杂、人工排查困难、标注规范不统一等痛点。

---

## 🔑 核心特性

### 🔄 多模态/多格式适配器
支持五种主流数据格式：
- **HDF5** (工业级科学数据)
- **ROS/mcap** (机器人操作系统)
- **LeRobot** (开源机器人框架)
- **Unitree** (商用机器人SDK)
- **Folder** (图片序列)

通过工厂模式统一接口，新增格式只需实现`Adapter`接口。现支持基于JSON规则的动态字段映射（Dynamic Key Mapping）：
```python
class BaseDatasetReader(ABC):
    def __init__(self, config: Optional[AdapterConfig] = None): ...
    @abstractmethod
    def load(self, file_path: str) -> bool: ...
```

### 🚀 基于 Rerun 的极速审查
- **自动抽样对比**：提取首/中/尾帧生成对比视图
- **交互式审核**：`B`键标记异常数据自动隔离至`_QUARANTINE`文件夹
- **性能优势**：Rerun引擎实现1000FPS级帧率渲染

### 🎛️ 动态配置驱动的标注UI
基于`configs/vocabulary.json`驱动的声明式表单：
- 支持**级联下拉框**（如场景分类）
- 动态表格组件（物品属性标注）
- 实时YAML生成

示例字段定义：
```json
{
  "key": "scene_level1",
  "type": "selectbox",
  "options": {
    "Retail": "商超零售",
    "Catering": "餐饮服务"
  }
}
```

### 🤖 内置大模型服务
集成阿里云Qwen接口：
- 中文指令自动翻译成专业英文
- 示例输入：
  ```text
  把篮子放在桌子中间对齐，然后把桃子放进篮子
  ```
- 输出结果：
  ```text
  Align the basket at the center of the table, then place the peaches into the basket with precise orientation control
  ```

---

## ⚙️ 安装指南
使用uv进行依赖管理（推荐清华源）：
```bash
# 安装uv
pip install uv
# 安装系统依赖
sudo apt-get install python3-tk
# 同步依赖
uv sync

# 启动标注系统
uv run streamlit run src/ui/annotation_app.py

# 安装为系统工具
uv install -e .
```

依赖列表：
| 模块          | 用途                  |
|---------------|-----------------------|
| h5py          | HDF5格式处理          |
| rerun-sdk     | 三维可视化引擎        |
| rosbags       | ROS数据解析           |
| pyyaml        | YAML配置生成          |
| openai        | 大模型服务接口        |

---

## 🚀 快速开始

### 第一步：数据清洗与排查
1. 启动可视化审查
   ```bash
   uv run python src/core/reviewer.py --data_path ./dataset
   ```
2. 使用`B`键标记异常样本
3. 查看隔离文件夹`./dataset/_QUARANTINE`

### 第二步：元数据标注
1. 启动Web标注界面
   ```bash
   uv run streamlit run src/ui/annotation_app.py
   ```
2. 选择数据集目录
3. 填写动态表单（支持字段自动补全）
4. 导出YAML配置文件

---

## 🧠 高级配置
### 底层数据解析规则 (Adapter Rules)
可通过修改 `configs/adapter_rules.json` 无代码接入全新机器人数据格式，无需修改核心Python代码。

示例配置：
```json
{
  "Custom_ROS_Robot": {
    "base_type": "ROS",
    "match_rules": {
      "path_keywords": ["CustomRobot"]
    },
    "image_keys_map": {
      "cam_front": "/camera_front/color/image_raw",
      "cam_wrist": "/camera_wrist/color/image_raw"
    }
  }
}
```

### 添加新场景分类
在`configs/vocabulary.json`中修改：
```json
{
  "key": "scene_level1",
  "options": {
    "Retail": "商超零售",
    "Catering": "餐饮服务",
    "NewScene": "新场景类型"  // ← 新增选项
  }
}
```

### 定义级联字段
```json
{
  "key": "scene_level2",
  "type": "selectbox_dependent",
  "depends_on": "scene_level1",
  "options_map": {
    "NewScene": {
      "SubScene1": "子场景1",
      "SubScene2": "子场景2"
    }
  }
}
```

---

## 📁 目录结构
```
robocoin-viewer/
├── configs/              # 配置文件
│   ├── vocabulary.json   # 标注字段定义
│   ├── adapter_rules.json # 底层数据解析与字段映射规则
│   └── schemas/          # 数据校验规则
├── src/
│   ├── adapters/         # 数据格式适配器
│   ├── core/             # 核心逻辑
│   │   ├── factory.py    # 工厂模式实现
│   │   └── llm_service.py# 大模型服务
│   └── ui/               # Web界面
│       └── annotation_app.py # Streamlit主程序
├── tools/                # 辅助工具
└── tests/                # 单元测试
```

---
