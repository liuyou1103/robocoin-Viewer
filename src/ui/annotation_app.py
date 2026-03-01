import sys
import os
import json
import time
import tkinter as tk
from tkinter import filedialog
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

def open_file_dialog(is_dir=False, filetypes=[("All files", "*.*")]):
    """弹出系统原生文件/文件夹选择对话框"""
    try:
        root = tk.Tk()
        root.withdraw() # 隐藏主窗口
        root.wm_attributes('-topmost', 1) # 强制对话框在最顶层
        if is_dir:
            path = filedialog.askdirectory(master=root, title="选择文件夹")
        else:
            path = filedialog.askopenfilename(master=root, title="选择文件", filetypes=filetypes)
        root.destroy()
        return path
    except Exception as e:
        st.error(f"❌ 无法呼出文件选择器 (如果在纯命令行的 SSH 下运行不支持弹出界面): {e}")
        return ""

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
    """从 'English (中文)' 格式中提取 'English'"""
    if isinstance(val, str) and " (" in val and val.endswith(")"):
        return val.split(" (")[0]
    return val

def render_field(field, current_data, all_fields=None):
    """配置驱动：根据 schema 定义自动渲染 Streamlit 控件并返回收集到的值"""
    key = field["key"]
    label = field["label"]
    ftype = field["type"]
    
    if ftype == "text":
        return st.text_input(label, placeholder=field.get("placeholder", ""))
        
    # 👇 新增：支持 Qwen 翻译的【复合数据集名称控件】 👇
    elif ftype == "dataset_name_builder":
        # 1. 获取设备选项（代码逻辑保持不变）
        device_opts = {}
        if all_fields:
            for f in all_fields:
                if f.get("key") == "device_model":
                    device_opts = f.get("options", {})
                    break
        
        st.markdown(f"**{label}**")
        
        # 2. 初始化 state 存储后缀
        state_key = f"suffix_val_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = ""

        # 🌟 定义翻译的回调函数
        def translate_callback():
            # 回调函数在脚本重新运行前执行，此时修改 state 是安全的
            api_key = st.session_state.get("dashscope_api_key", "")
            current_text = st.session_state[state_key] # 直接从绑定的 key 中取值
            
            if api_key and current_text.strip():
                try:
                    from src.core.llm_service import QwenLLMService
                    # 临时注入环境变量给 LLMService 使用
                    os.environ["DASHSCOPE_API_KEY"] = api_key
                    llm = QwenLLMService()
                    translated = llm.translate_task_name(current_text)
                    # 更新 state，下次 Rerun 时输入框会自动显示新值
                    st.session_state[state_key] = translated
                except Exception as e:
                    # 注意：回调函数里没法直接 st.error，通常用 print 或稍后在 UI 处理
                    print(f"翻译回调失败: {e}")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            opts_keys = list(device_opts.keys())
            fmt_func = lambda x, d=device_opts: f"{x} ({d[x]})" if d.get(x) else x
            prefix = st.selectbox("1. 选择本体型号", options=opts_keys, format_func=fmt_func, key=f"prefix_{key}")
            
        with col2:
            # 🌟 关键：直接绑定 key，不要设置 value 参数
            st.text_input(
                "2. 动词_名词 (可输中文)", 
                placeholder="例如: pick_apple", 
                key=state_key 
            )
            
        with col3:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            # 🌟 关键：使用 on_click 触发回调
            st.button(
                "🌐 翻译名称", 
                key=f"trans_name_btn_{key}", 
                on_click=translate_callback,
                help="点击将中文动作描述翻译为 verb_noun 格式"
            )
            
        # 拼接最终的名字（直接从 state 读最新值）
        final_name = f"{prefix}_{st.session_state[state_key]}" if st.session_state[state_key] else prefix
        st.caption(f"预览最终名称: `{final_name}`")
        return final_name
        
    elif ftype == "textarea":
        state_key = f"textarea_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = field.get("default", "")
            
        val = st.text_area(label, value=st.session_state[state_key], height=100)
        st.session_state[state_key] = val
        
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
                            os.environ["DASHSCOPE_API_KEY"] = api_key 
                            from src.core.llm_service import QwenLLMService
                            llm = QwenLLMService()
                            lines = [i.strip() for i in val.split('\n') if i.strip()]
                            translated_lines = llm.translate_instructions(lines)
                            st.session_state[state_key] = "\n".join(translated_lines)
                            st.rerun()
                        except Exception as e:
                            st.error(f"翻译失败: {e}")
                            
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
    columns = []
    for idx, name in enumerate(sample_names):
        cam_views = []
        for cam in cameras:
            cam_views.append(rrb.Spatial2DView(
                origin=f"preview/sample_{idx}/{cam}",
                name=f"{cam}"
            ))
        
        cam_grid = rrb.Grid(*cam_views)
        columns.append(rrb.Vertical(
            rrb.TextDocumentView(origin=f"preview/sample_{idx}/info", name=f"{name}"),
            cam_grid,
            row_shares=[1, 12], 
            name=f"Sample {idx+1}"
        ))
        
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
    
    # 初始化 Session State 中的默认路径
    if 'vocab_path' not in st.session_state:
        st.session_state['vocab_path'] = os.path.abspath("./configs/vocabulary.json")
    if 'dataset_path' not in st.session_state:
        st.session_state['dataset_path'] = os.path.abspath("./data/hdf5")

    # 侧边栏：配置区
    with st.sidebar:
        st.header("🔑 大模型服务配置")
        api_key_input = st.text_input(
            "DashScope API Key", 
            value=os.environ.get("DASHSCOPE_API_KEY", ""),
            type="password",
            help="输入阿里云百炼的 API Key，用于自动翻译。如果在系统中已配置环境变量，此处会自动读取。"
        )
        st.session_state["dashscope_api_key"] = api_key_input
        st.markdown("---")

        st.header("📂 数据源与配置项")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            vocab_input = st.text_input("Schema 配置文件 (JSON):", value=st.session_state['vocab_path'])
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📁", key="btn_vocab", help="点击浏览选择 JSON 文件"):
                p = open_file_dialog(is_dir=False, filetypes=[("JSON Files", "*.json")])
                if p:
                    st.session_state['vocab_path'] = p
                    st.rerun()
        if vocab_input != st.session_state['vocab_path']:
            st.session_state['vocab_path'] = vocab_input
            st.rerun()

        vocab = load_vocabulary(st.session_state['vocab_path'])

        col3, col4 = st.columns([4, 1])
        with col3:
            dataset_input = st.text_input("输入数据集根路径:", value=st.session_state['dataset_path'])
        with col4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📁", key="btn_dataset", help="点击浏览选择包含数据的文件夹"):
                p = open_file_dialog(is_dir=True)
                if p:
                    st.session_state['dataset_path'] = p
                    st.rerun()
        if dataset_input != st.session_state['dataset_path']:
            st.session_state['dataset_path'] = dataset_input
            st.rerun()
            
        target_dir = st.session_state['dataset_path']

        if st.button("1. 扫描与类型检查", type="primary"):
            if not os.path.exists(target_dir):
                st.error("路径不存在！")
            else:
                inspector = DatasetInspector(target_dir)
                inspector.scan()
                if inspector.check_consistency():
                    st.session_state['grouped_datasets'] = inspector.grouped_datasets
                    st.session_state['valid_paths'] = inspector.get_all_valid_paths()
                    st.success("扫描成功！")
                else:
                    st.error("一致性检查失败，请检查数据格式。")
        
        if 'valid_paths' in st.session_state:
            st.info(f"当前有效数据总量: {len(st.session_state['valid_paths'])} 条")

    # 主界面分为三个 Tab
    tab1, tab2, tab3 = st.tabs(["🔍 第一步：数据清洗与排查", "📝 第二步：元数据标注", "⚙️ 第三步：词库维护 (Schema)"])

    # ==========================================
    # TAB 1: 数据清洗与预览
    # ==========================================
    with tab1:
        if 'grouped_datasets' not in st.session_state:
            st.info("👈 请先在左侧选择路径并点击「扫描与类型检查」")
        else:
            grouped_datasets = st.session_state['grouped_datasets']
            valid_paths = st.session_state['valid_paths']
            
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
    # TAB 2: 元数据标注 (生成 YAML)
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
                        collected_data[field["key"]] = render_field(field, collected_data, schema_fields)
                        
        with col2:
            for g in col2_groups:
                if g in groups:
                    st.subheader(g)
                    for field in groups[g]:
                        collected_data[field["key"]] = render_field(field, collected_data, schema_fields)

        st.markdown("---")
        if st.button("💾 生成 YAML 标注文件", type="primary"):
            if 'dataset_path' not in st.session_state:
                st.warning("请先在「数据清洗与排查」页面加载数据！")
            elif not collected_data.get("dataset_name"):
                st.error("请填写数据集名称！")
            else:
                save_path = ConfigGenerator.analyze_and_save(collected_data, target_dir, filename="local_dataset_info.yaml")
                st.success(f"🎉 标注文件生成成功！\n文件路径: `{save_path}`")
                
                with st.expander("点击查看生成的 YAML 内容 (纯英文)"):
                    st.code(ConfigGenerator.generate_yaml_string(collected_data), language="yaml")

    # ==========================================
    # TAB 3: 词库维护 (可视化配置)
    # ==========================================
    with tab3:
        st.header("⚙️ 可视化词库与表单维护")
        curr_vocab_path = st.session_state.get('vocab_path', '')
        st.markdown(f"当前正在维护的 Schema: `{curr_vocab_path}`")
        
        if not os.path.exists(curr_vocab_path):
            st.error("指定的 Schema 文件不存在，请先在侧边栏选择有效的 JSON 文件。")
        else:
            try:
                with open(curr_vocab_path, 'r', encoding='utf-8') as f:
                    current_schema = json.load(f)
                fields = current_schema.get("fields", [])
            except Exception as e:
                st.error(f"无法读取或解析 Schema 文件: {e}")
                fields = []

            if fields:
                st.info("💡 **提示**: 直接在下方对应字段中添加或修改选项。所有的更改都会直接写入 JSON，并立刻在第二步标注界面生效。")
                
                for idx, field in enumerate(fields):
                    field_type = field.get("type", "")
                    
                    if field_type in ["selectbox", "multiselect"]:
                        with st.expander(f"📝 字段: {field['label']} ({field['key']})"):
                            st.markdown(f"**所属分组:** {field.get('group', '未分组')} | **类型:** {field_type}")
                            options = field.get("options", {})
                            
                            if options:
                                st.markdown("##### 现有选项")
                                opt_cols = st.columns(3)
                                for i, (opt_key, opt_label) in enumerate(list(options.items())):
                                    col = opt_cols[i % 3]
                                    col.markdown(f"`{opt_key}` - {opt_label}")
                                    if col.button(f"🗑️ 删除", key=f"del_{field['key']}_{opt_key}", help=f"删除 {opt_key}"):
                                        del current_schema["fields"][idx]["options"][opt_key]
                                        with open(curr_vocab_path, 'w', encoding='utf-8') as f:
                                            json.dump(current_schema, f, ensure_ascii=False, indent=2)
                                        load_vocabulary.clear()
                                        st.rerun()
                            else:
                                st.caption("暂无选项")
                                
                            st.markdown("##### ➕ 新增选项")
                            new_col1, new_col2, new_col3 = st.columns([2, 2, 1])
                            with new_col1: new_opt_key = st.text_input("英文 Key (如 apple)", key=f"new_key_{field['key']}")
                            with new_col2: new_opt_label = st.text_input("中文显示名 (如 苹果)", key=f"new_label_{field['key']}")
                            with new_col3:
                                st.markdown("<br>", unsafe_allow_html=True) 
                                if st.button("添加", key=f"add_{field['key']}", use_container_width=True):
                                    if not new_opt_key.strip() or not new_opt_label.strip():
                                        st.warning("不能为空！")
                                    elif new_opt_key in options:
                                        st.warning(f"'{new_opt_key}' 已存在！")
                                    else:
                                        if "options" not in current_schema["fields"][idx]:
                                            current_schema["fields"][idx]["options"] = {}
                                        current_schema["fields"][idx]["options"][new_opt_key.strip()] = new_opt_label.strip()
                                        with open(curr_vocab_path, 'w', encoding='utf-8') as f:
                                            json.dump(current_schema, f, ensure_ascii=False, indent=2)
                                        load_vocabulary.clear()
                                        st.rerun()

                    elif field_type == "selectbox_dependent":
                        with st.expander(f"🔗 级联字段: {field['label']} ({field['key']} 依赖于 {field.get('depends_on', '未知')})"):
                            options_map = field.get("options_map", {})
                            parent_key = field.get("depends_on")
                            
                            # --- 核心改进：获取父级字段真正拥有的所有选项 ---
                            parent_options = {}
                            for f in fields:
                                if f.get("key") == parent_key:
                                    parent_options = f.get("options", {})
                                    break
                            
                            # 所有的父级 Key (来自父级字段的定义)
                            all_valid_parent_keys = list(parent_options.keys())
                            
                            st.markdown("##### 现有选项映射")
                            if options_map:
                                for p_val, sub_options in list(options_map.items()):
                                    # 如果父级 Key 在父级字段里已经被删了，这里用红色标出提醒
                                    label_suffix = "" if p_val in parent_options else " ⚠️ (父级已不存在)"
                                    st.markdown(f"**当父级选择 `{p_val}` ({parent_options.get(p_val, '未知')}){label_suffix} 时：**")
                                    
                                    sub_cols = st.columns(3)
                                    for i, (opt_key, opt_label) in enumerate(list(sub_options.items())):
                                        col = sub_cols[i % 3]
                                        col.markdown(f"`{opt_key}` - {opt_label}")
                                        if col.button("🗑️", key=f"del_sub_{field['key']}_{p_val}_{opt_key}"):
                                            del current_schema["fields"][idx]["options_map"][p_val][opt_key]
                                            # 如果该父级下没有子项了，连父级 Key 一起删掉，保持 JSON 干净
                                            if not current_schema["fields"][idx]["options_map"][p_val]:
                                                del current_schema["fields"][idx]["options_map"][p_val]
                                            with open(curr_vocab_path, 'w', encoding='utf-8') as f:
                                                json.dump(current_schema, f, ensure_ascii=False, indent=2)
                                            load_vocabulary.clear()
                                            st.rerun()
                            else:
                                st.caption("暂无映射关系")
                                        
                            st.markdown("---")
                            st.markdown("##### ➕ 为父级新增子选项")
                            if not all_valid_parent_keys:
                                st.warning(f"请先在「{parent_key}」字段中添加选项！")
                            else:
                                new_col1, new_col2, new_col3, new_col4 = st.columns([2, 2, 2, 1])
                                
                                with new_col1:
                                    # 这里保证了下拉列表里一定会出现你在“一级场景”新加的东西
                                    target_p = st.selectbox(
                                        "选择父级 Key", 
                                        all_valid_parent_keys, 
                                        format_func=lambda x: f"{x} ({parent_options.get(x)})",
                                        key=f"target_p_sel_{field['key']}"
                                    )
                                with new_col2:
                                    new_s_key = st.text_input("子项英文 Key", key=f"new_s_k_{field['key']}")
                                with new_col3:
                                    new_s_label = st.text_input("子项中文名", key=f"new_s_l_{field['key']}")
                                with new_col4:
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    if st.button("添加", key=f"add_sub_btn_{field['key']}", use_container_width=True):
                                        if new_s_key.strip() and new_s_label.strip():
                                            if "options_map" not in current_schema["fields"][idx]:
                                                current_schema["fields"][idx]["options_map"] = {}
                                            
                                            # 初始化该父级的字典
                                            if target_p not in current_schema["fields"][idx]["options_map"]:
                                                current_schema["fields"][idx]["options_map"][target_p] = {}
                                                
                                            current_schema["fields"][idx]["options_map"][target_p][new_s_key.strip()] = new_s_label.strip()
                                            
                                            with open(curr_vocab_path, 'w', encoding='utf-8') as f:
                                                json.dump(current_schema, f, ensure_ascii=False, indent=2)
                                            load_vocabulary.clear()
                                            st.rerun()

if __name__ == "__main__":
    main()