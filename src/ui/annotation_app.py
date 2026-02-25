import sys
import os
import json
import time
import streamlit as st
import rerun as rr
import rerun.blueprint as rrb

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
def load_vocabulary():
    """读取外部词库文件"""
    vocab_path = os.path.join(os.path.dirname(__file__), '../../configs/vocabulary.json')
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"找不到词库文件: {vocab_path}")
        return {}

def setup_comparison_layout(sample_names, cameras):
    """配置 Rerun 并排对比视图的蓝图"""
    columns = []
    for idx, name in enumerate(sample_names):
        cam_views = []
        for cam in cameras:
            cam_views.append(rrb.Spatial2DView(
                origin=f"preview/sample_{idx}/{cam}",
                name=f"{cam}"
            ))
        columns.append(rrb.Vertical(
            rrb.TextDocumentView(origin=f"preview/sample_{idx}/info", name=f"{name}"),
            *cam_views,
            name=f"Sample {idx+1}"
        ))
    blueprint = rrb.Blueprint(rrb.Horizontal(*columns), collapse_panels=True)
    rr.send_blueprint(blueprint)

def run_parallel_preview(sample_paths):
    """执行多样本抽样预览播放逻辑"""
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
        
        # UI 进度条更新
        if i % 10 == 0 or i == max_len - 1:
            progress_bar.progress((i + 1) / max_len, text=f"播放进度: {i+1}/{max_len} 帧")
            
    for r in readers: r.close()
    progress_bar.empty()
    st.success("✅ 预览播放完成，请在 Rerun 窗口查看。")

def main():
    st.title("🤖 RoboCoin 数据集清洗与标注系统")
    vocab = load_vocabulary()
    
    # 侧边栏：全局路径配置
    with st.sidebar:
        st.header("📂 数据源配置")
        dataset_path = st.text_input("输入数据集根路径:", value="./data/hdf5")
        
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
    # TAB 1: 数据清洗与预览 (复刻 main.py 逻辑)
    # ==========================================
    with tab1:
        if 'grouped_datasets' not in st.session_state:
            st.info("👈 请先在左侧输入路径并点击「扫描与类型检查」")
        else:
            grouped_datasets = st.session_state['grouped_datasets']
            valid_paths = st.session_state['valid_paths']
            target_dir = st.session_state['dataset_path']
            
            # --- 步骤 1: 处理混入的不同类型数据 ---
            st.subheader("1. 数据类型检查与整理")
            if len(grouped_datasets) > 1:
                st.warning(f"检测到 {len(grouped_datasets)} 种混合的数据类型: {list(grouped_datasets.keys())}")
                if st.button("自动分类并物理隔离不同类型数据"):
                    organizer = DatasetOrganizer(target_dir)
                    new_grouped_paths = organizer.sort_by_type(grouped_datasets, target_dir)
                    st.session_state['grouped_datasets'] = new_grouped_paths
                    st.success("✅ 数据已按照类型分组移动到独立文件夹中。")
                    st.rerun()
                
                # 下拉选择要专注审阅的类型
                selected_type = st.selectbox("请选择本次要处理的数据类型:", list(grouped_datasets.keys()))
                valid_paths = grouped_datasets[selected_type]
                st.session_state['valid_paths'] = valid_paths
            else:
                st.success(f"✅ 数据类型一致，仅包含: {list(grouped_datasets.keys())[0]}")

            # --- 步骤 2: 人工审核（剔除其他任务数据） ---
            st.subheader("2. 人工逐帧审核 (排查不同任务数据)")
            st.markdown("通过 Rerun 窗口检查是否混入了**无关任务/动作错误**的数据。")
            if st.button("🚀 启动人工审核 (Rerun)"):
                with st.spinner("请在弹出的 Rerun 窗口中操作 (使用键盘 N/P 切换, Space 标记异常, Esc 退出)..."):
                    viz = RerunVisualizer("RoboCoin_Review")
                    reviewer = DatasetReviewer(viz)
                    bad_datasets = reviewer.start_review(valid_paths)
                    
                    if bad_datasets:
                        organizer = DatasetOrganizer(target_dir)
                        quarantine_dir = organizer.quarantine_bad_data(bad_datasets, target_dir)
                        st.warning(f"🔒 发现异常任务数据！已将其隔离到: {quarantine_dir}")
                        # 实时更新 session 中的好数据列表
                        final_paths = [p for p in valid_paths if p not in bad_datasets]
                        st.session_state['valid_paths'] = final_paths
                        # 如果需要可以调用原有的 save_report 逻辑
                        st.success(f"🧹 剔除异常数据后，剩余有效数据: {len(final_paths)} 条")
                    else:
                        st.success("✨ 完美！未发现混入的其他任务数据。")

            # --- 步骤 3: 抽样对比预览 ---
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("基本信息")
            dataset_name = st.text_input("数据集名称 (本体_动词_名词)", placeholder="Galaxea_R1_Lite_storage_peach")
            task_instruction_raw = st.text_area(
                "Task Instruction (任务指令, 每行一条)", 
                value="place the basket in the center of the table and align it, then put the peaches into the basket.",
                height=100
            )
            task_instructions = [i.strip() for i in task_instruction_raw.split('\n') if i.strip()]

            st.subheader("场景设置")
            env_type = st.selectbox("环境类型 (env_type)", vocab.get("env_type", []))
            
            scene_col1, scene_col2 = st.columns(2)
            with scene_col1:
                scene_level1 = st.selectbox("一级场景 (scene_level1)", vocab.get("scene_type_level1", []))
            with scene_col2:
                scene_level2 = st.selectbox("二级场景 (scene_level2)", vocab.get("scene_type_level2", []))

        with col2:
            st.subheader("动作与物品")
            atomic_actions = st.multiselect("原子动作 (atomic_actions)", vocab.get("atomic_actions", []))
            
            st.markdown("**操作物品 (Objects)**")
            if 'objects_list' not in st.session_state:
                st.session_state['objects_list'] = [{"name": "table", "color": "red"}]
                
            edited_objects = st.data_editor(
                st.session_state['objects_list'], 
                num_rows="dynamic",
                column_config={
                    "name": st.column_config.SelectboxColumn("物品名称", options=vocab.get("object_names", []), required=True),
                    "color": st.column_config.SelectboxColumn("颜色", options=vocab.get("colors", []), required=True)
                },
                use_container_width=True
            )

            st.subheader("硬件配置")
            op_height = st.number_input("操作台高度 (operation_platform_height)", value=77.2, format="%.1f")
            device_models = st.multiselect("本体型号 (device_model)", vocab.get("device_models", []))
            end_effector_types = st.multiselect("末端执行器 (end_effector_type)", vocab.get("end_effector_types", []))
            task_operation_type = st.selectbox("操作类型 (task_operation_type)", vocab.get("task_operation_types", []))
            tele_type = st.selectbox("遥操作方式 (tele_type)", vocab.get("tele_types", []))

        st.markdown("---")
        if st.button("💾 生成 YAML 标注文件", type="primary"):
            if 'dataset_path' not in st.session_state:
                st.warning("请先在「数据清洗与排查」页面加载数据！")
            elif not dataset_name:
                st.error("请填写数据集名称！")
            else:
                data_dict = {
                    "dataset_name": dataset_name,
                    "dataset_uuid": "", 
                    "task_instruction": task_instructions,
                    "env_type": env_type,
                    "scene_level1": scene_level1,
                    "scene_level2": scene_level2,
                    "atomic_actions": atomic_actions,
                    "objects": [{"object_name": obj["name"], "color": obj["color"]} for obj in edited_objects if obj.get("name")],
                    "operation_platform_height": op_height,
                    "device_model": device_models,
                    "end_effector_type": end_effector_types,
                    "task_operation_type": task_operation_type,
                    "tele_type": tele_type
                }
                
                save_path = ConfigGenerator.analyze_and_save(data_dict, st.session_state['dataset_path'])
                st.success(f"🎉 标注文件生成成功！\n文件路径: `{save_path}`")
                
                with st.expander("点击查看生成的 YAML 内容"):
                    st.code(ConfigGenerator.generate_yaml_string(data_dict), language="yaml")

if __name__ == "__main__":
    main()