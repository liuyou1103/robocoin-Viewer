import sys
import os
import json
import time
import streamlit as st
import rerun as rr
import rerun.blueprint as rrb
from openai import OpenAI

# 确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.inspector import DatasetInspector
from src.core.organizer import DatasetOrganizer
from src.core.reviewer import DatasetReviewer
from src.core.factory import ReaderFactory
from src.core.config_generator import ConfigGenerator
from src.ui.rerun_visualizer import RerunVisualizer

# 页面配置
st.set_page_config(page_title="RoboCoin Annotation Tool", layout="wide")

@st.cache_data
def load_vocabulary(vocab_path):
    """读取外部词库文件 (Schema 配置)"""
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ 找不到 Schema 配置文件: {vocab_path} (请在左侧侧边栏确认路径)")
        return {"fields": []}
    except json.JSONDecodeError:
        st.error(f"❌ 配置文件格式错误，请检查 {vocab_path} 是否为合法的 JSON！")
        return {"fields": []}

def clean_editor_value(val):
    """
    从 'English (中文)' 格式中提取 'English'
    用于清洗 data_editor 的返回值
    """
    if isinstance(val, str) and " (" in val and val.endswith(")"):
        return val.split(" (")[0]
    return val

def render_field(field, current_data):
    """配置驱动：根据 schema 定义自动渲染 Streamlit 控件并返回收集到的值"""
    key = field["key"]
    label = field["label"]
    ftype = field["type"]
    
    if ftype == "text":
        return st.text_input(label, placeholder=field.get("placeholder", ""))
        
    elif ftype == "textarea":
        # 引入 state 机制以便翻译后覆盖当前文本
        state_key = f"textarea_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = field.get("default", "")
            
        val = st.text_area(label, value=st.session_state[state_key], height=100)
        st.session_state[state_key] = val
        
        # 👇 为任务指令专属定制的一键翻译挂载逻辑 👇
        if key == "task_instruction":
            if st.button("🌐 使用 Qwen 一键翻译为专业英文", type="secondary"):
                api_key = st.session_state.get("dashscope_api_key", os.environ.get("DASHSCOPE_API_KEY", ""))
                if not api_key:
                    st.error("⚠️ 请先在左侧侧边栏配置阿里云 API Key！")
                elif not val.strip():
                    st.warning("请先输入需要翻译的指令！")
                else:
                    with st.spinner("正在请求大模型翻译，请稍候..."):
                        try:
                            # 动态注入环境变量给 LLMService 使用
                            os.environ["DASHSCOPE_API_KEY"] = api_key 
                            from src.core.llm_service import QwenLLMService
                            llm = QwenLLMService()
                            lines = [i.strip() for i in val.split('\n') if i.strip()]
                            translated_lines = llm.translate_instructions(lines)
                            
                            # 覆盖 session state 并刷新页面
                            st.session_state[state_key] = "\n".join(translated_lines)
                            st.rerun()
                        except Exception as e:
                            st.error(f"翻译失败: {e}")
        # 👆 ------------------------------------------ 👆
                            
        return [i.strip() for i in val.split('\n') if i.strip()] 
        
    elif ftype == "selectbox":
        opts_dict = field.get("options", {})
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) else x
        return st.selectbox(label, options=opts_keys, format_func=fmt_func)

    elif ftype == "selectbox_dependent":
        parent_key = field.get("depends_on")
        parent_val = current_data.get(parent_key) 
        
        opts_map = field.get("options_map", {})
        opts_dict = opts_map.get(parent_val, {"unknown": "请先选择上级或暂无选项"})
        
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) and x != "unknown" else x
        return st.selectbox(label, options=opts_keys, format_func=fmt_func)
        
    elif ftype == "multiselect":
        opts_dict = field.get("options", {})
        opts_keys = list(opts_dict.keys())
        fmt_func = lambda x, d=opts_dict: f"{x} ({d[x]})" if d.get(x) else x
        return st.multiselect(label, options=opts_keys, format_func=fmt_func)
        
    elif ftype == "number":
        return st.number_input(label, value=field.get("default", 0.0), format="%.1f")
        
    elif ftype == "object_table":
        if f'table_{key}' not in st.session_state:
            st.session_state[f'table_{key}'] = [{"object_name": "table", "color": "red"}]
            
        name_opts = field.get("name_options", {})
        color_opts = field.get("color_options", {})
        name_display = [f"{k} ({v})" for k, v in name_opts.items()]
        color_display = [f"{k} ({v})" for k, v in color_opts.items()]
        
        edited = st.data_editor(
            st.session_state[f'table_{key}'],
            num_rows="dynamic",
            column_config={
                "object_name": st.column_config.SelectboxColumn("物品名称", options=name_display, required=True),
                "color": st.column_config.SelectboxColumn("颜色", options=color_display, required=True)
            },
            width="stretch"
        )
        
        cleaned = []
        for row in edited:
            if row.get("object_name"):
                cleaned.append({
                    "object_name": clean_editor_value(row["object_name"]),
                    "color": clean_editor_value(row["color"])
                })
        return cleaned

def setup_comparison_layout(sample_names, cameras):
    """配置 Rerun 并排对比视图的蓝图 (自适应网格排版优化版)"""
    columns = []
    for idx, name in enumerate(sample_names):
        cam_views = []
        for cam in cameras:
            cam_views.append(rrb.Spatial2DView(
                origin=f"preview/sample_{idx}/{cam}",
                name=f"{cam}"
            ))
        
        # 👇 核心改动：把该样本下的所有相机视角，丢进自适应网格中 👇
        # Rerun 会根据相机数量(比如6个)自动排版成完美的 3x2 或 2x3 网格
        cam_grid = rrb.Grid(*cam_views)
        
        # 把文本框和相机网格垂直组合
        columns.append(rrb.Vertical(
            rrb.TextDocumentView(origin=f"preview/sample_{idx}/info", name=f"{name}"),
            cam_grid,
            row_shares=[1, 12], # 比例分配：文字占 1 份，画面网格占 12 份，大幅压缩文字框高度
            name=f"Sample {idx+1}"
        ))
        
    # 整体水平排布多个样本 (首、中、尾)
    blueprint = rrb.Blueprint(rrb.Horizontal(*columns), collapse_panels=True)
    rr.send_blueprint(blueprint)

def run_parallel_preview(sample_paths):
    temp_reader = ReaderFactory.get_reader(sample_paths[0])
    temp_reader.load(sample_paths[0])
    cameras = temp_reader.get_all_sensors()
    
    rr.init("RoboCoin_Preview", spawn=True)
    setup_comparison_layout([os.path.basename(p) for p in sample_paths], cameras)
    temp_reader.close()
    
    rr.log("preview", rr.Clear(recursive=True))
    
    readers = []
    max_len = 0
    for p in sample_paths:
        r = ReaderFactory.get_reader(p)
        r.load(p)
        readers.append(r)
        if r.get_length() > max_len: 
            max_len = r.get_length()
    
    progress_bar = st.progress(0, text="正在同步播放视频流...")
    for i in range(max_len):
        rr.set_time_sequence("frame_idx", i)
        for s_idx, r in enumerate(readers):
            if i >= r.get_length(): continue
            frame = r.get_frame(i)
            for cam, img in frame.images.items():
                rr.log(f"preview/sample_{s_idx}/{cam}", rr.Image(img))
            if i == 0:
                rr.log(f"preview/sample_{s_idx}/info", rr.TextDocument(f"### {os.path.basename(sample_paths[s_idx])}"))
        
        if i % 10 == 0 or i == max_len - 1:
            progress_bar.progress((i + 1) / max_len, text=f"播放进度: {i+1}/{max_len} 帧")
            
    for r in readers: r.close()
    progress_bar.empty()
    st.success("✅ 预览播放完成，请在 Rerun 窗口查看。")

def main():
    st.title("🤖 RoboCoin 数据集清洗与标注系统")
    
    # 侧边栏：配置区
    with st.sidebar:
        # 👇 新增：大模型参数配置区 👇
        st.header("🔑 大模型服务配置")
        api_key_input = st.text_input(
            "DashScope API Key", 
            value=os.environ.get("DASHSCOPE_API_KEY", ""),
            type="password",
            help="输入阿里云百炼的 API Key，用于自动翻译。如果在系统中已配置环境变量，此处会自动读取。"
        )
        # 将配置保存至全局 session_state 中
        st.session_state["dashscope_api_key"] = api_key_input
        st.markdown("---")
        # 👆 ---------------------- 👆

        st.header("📂 数据源与配置项")
        
        # 1. 新增 Schema JSON 路径输入框 (默认指向项目的 configs 目录)
        vocab_path = st.text_input("Schema 配置文件路径 (JSON):", value="./configs/vocabulary.json")
        
        # 2. 数据集路径输入框
        dataset_path = st.text_input("输入数据集根路径:", value="./data/hdf5")
        
        # 3. 根据用户输入的路径，动态加载并渲染词库
        vocab = load_vocabulary(vocab_path)
        
        if st.button("1. 扫描与类型检查", type="primary"):
            if not os.path.exists(dataset_path):
                st.error("路径不存在！")
            else:
                inspector = DatasetInspector(dataset_path)
                inspector.scan()
                if inspector.check_consistency():
                    st.session_state['grouped_datasets'] = inspector.grouped_datasets
                    st.session_state['dataset_path'] = dataset_path
                    st.session_state['valid_paths'] = inspector.get_all_valid_paths()
                    st.success("扫描成功！")
                else:
                    st.error("一致性检查失败，请检查数据格式。")
        
        if 'valid_paths' in st.session_state:
            st.info(f"当前有效数据总量: {len(st.session_state['valid_paths'])} 条")

    # 主界面分为两个 Tab
    tab1, tab2 = st.tabs(["🔍 第一步：数据清洗与排查", "📝 第二步：元数据标注"])

    # ==========================================
    # TAB 1: 数据清洗与预览
    # ==========================================
    with tab1:
        if 'grouped_datasets' not in st.session_state:
            st.info("👈 请先在左侧输入路径并点击「扫描与类型检查」")
        else:
            grouped_datasets = st.session_state['grouped_datasets']
            valid_paths = st.session_state['valid_paths']
            target_dir = st.session_state['dataset_path']
            
            st.subheader("1. 数据类型检查与整理")
            if len(grouped_datasets) > 1:
                st.warning(f"检测到 {len(grouped_datasets)} 种混合的数据类型: {list(grouped_datasets.keys())}")
                if st.button("自动分类并物理隔离不同类型数据"):
                    organizer = DatasetOrganizer(target_dir)
                    new_grouped_paths = organizer.sort_by_type(grouped_datasets, target_dir)
                    st.session_state['grouped_datasets'] = new_grouped_paths
                    st.success("✅ 数据已按照类型分组移动到独立文件夹中。")
                    st.rerun()
                
                selected_type = st.selectbox("请选择本次要处理的数据类型:", list(grouped_datasets.keys()))
                valid_paths = grouped_datasets[selected_type]
                st.session_state['valid_paths'] = valid_paths
            else:
                st.success(f"✅ 数据类型一致，仅包含: {list(grouped_datasets.keys())[0]}")

            st.subheader("2. 人工逐帧审核 (排查不同任务数据)")
            st.markdown("通过 Rerun 窗口检查是否混入了**无关任务/动作错误**的数据。")
            if st.button("🚀 启动人工审核 (Rerun)"):
                with st.spinner("请在弹出的 Rerun 窗口中操作 (使用键盘 N/P 切换, B 标记异常, Q/Esc 退出)..."):
                    viz = RerunVisualizer("RoboCoin_Review")
                    reviewer = DatasetReviewer(viz)
                    bad_datasets = reviewer.start_review(valid_paths)
                    
                    if bad_datasets:
                        organizer = DatasetOrganizer(target_dir)
                        quarantine_dir = organizer.quarantine_bad_data(bad_datasets, target_dir)
                        st.warning(f"🔒 发现异常任务数据！已将其隔离到: {quarantine_dir}")
                        final_paths = [p for p in valid_paths if p not in bad_datasets]
                        st.session_state['valid_paths'] = final_paths
                        st.success(f"🧹 剔除异常数据后，剩余有效数据: {len(final_paths)} 条")
                        st.rerun()  
                    else:
                        st.success("✨ 完美！未发现混入的其他任务数据。")
                        st.rerun()

            st.subheader("3. 抽样对比预览 (检查命名与视频内容)")
            st.markdown("自动抽取 首、中、尾 3个样本进行并排播放，以便宏观确认数据与标注预期是否相符。")
            if st.button("📺 启动多视角对比预览"):
                if len(valid_paths) == 0:
                    st.error("没有可用数据进行预览。")
                else:
                    indices = [0]
                    if len(valid_paths) > 1: indices.append(len(valid_paths)-1)
                    if len(valid_paths) > 2: indices.insert(1, len(valid_paths)//2)
                    sample_paths = [valid_paths[i] for i in indices]
                    run_parallel_preview(sample_paths)

    # ==========================================
    # TAB 2: 元数据标注 (生成 YAML) - 完全动态驱动版
    # ==========================================
    with tab2:
        st.header("任务元数据标注")
        
        schema_fields = vocab.get("fields", [])
        groups = {}
        for f in schema_fields:
            g = f.get("group", "其他配置")
            if g not in groups: groups[g] = []
            groups[g].append(f)
            
        collected_data = {}
        col1, col2 = st.columns(2)
        
        col1_groups = ["基本信息", "场景设置"]
        col2_groups = ["动作与物品", "硬件配置", "其他配置"]
        
        all_preset_groups = set(col1_groups + col2_groups)
        new_groups = [g for g in groups.keys() if g not in all_preset_groups]
        
        for i, g in enumerate(new_groups):
            if i % 2 == 0:
                col1_groups.append(g)
            else:
                col2_groups.append(g)

        with col1:
            for g in col1_groups:
                if g in groups:
                    st.subheader(g)
                    for field in groups[g]:
                        collected_data[field["key"]] = render_field(field, collected_data)
                        
        with col2:
            for g in col2_groups:
                if g in groups:
                    st.subheader(g)
                    for field in groups[g]:
                        collected_data[field["key"]] = render_field(field, collected_data)

        st.markdown("---")
        if st.button("💾 生成 YAML 标注文件", type="primary"):
            if 'dataset_path' not in st.session_state:
                st.warning("请先在「数据清洗与排查」页面加载数据！")
            elif not collected_data.get("dataset_name"):
                st.error("请填写数据集名称！")
            else:
                save_path = ConfigGenerator.analyze_and_save(collected_data, st.session_state['dataset_path'])
                st.success(f"🎉 标注文件生成成功！\n文件路径: `{save_path}`")
                
                with st.expander("点击查看生成的 YAML 内容 (纯英文)"):
                    st.code(ConfigGenerator.generate_yaml_string(collected_data), language="yaml")

if __name__ == "__main__":
    main()