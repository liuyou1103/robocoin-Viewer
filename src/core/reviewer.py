# src/core/reviewer.py
import time
import threading
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb
from pynput import keyboard
from src.core.factory import ReaderFactory

class DatasetReviewer:
    def __init__(self, visualizer):
        """
        :param visualizer: RerunVisualizer 实例
        """
        self.viz = visualizer
        self.bad_datasets = [] 
        self.current_idx = 0            # 当前数据集文件夹的索引
        self.current_ep_idx = 0         # 新增：当前数据集内部的 Episode 索引
        self.total_episodes = 1         # 新增：当前数据集的总 Episode 数量
        
        self.current_path = None        # 当前正在审核的文件夹路径
        self.current_reader = None      # 驻留的 Reader 实例，避免重复加载
        
        self.dataset_paths = []
        self.is_running = False
        self.needs_refresh = False

    def start_review(self, dataset_paths: list):
        """
        启动交互式审核流程 (全局键盘监听版)
        """
        if not dataset_paths:
            print("❌ 没有数据可审核")
            return []

        self.dataset_paths = dataset_paths
        self.current_idx = 0
        self.current_ep_idx = 0
        self.is_running = True
        
        # 初始化加载第一个数据集的 Reader
        self._load_reader(self.dataset_paths[self.current_idx])
        self.needs_refresh = True

        print("\n" + "="*50)
        print("🕵️‍♂️ 进入交互式审核模式 (键盘控制)")
        print("="*50)
        print("保持 Rerun 窗口在前台即可，使用以下按键控制：")
        print("  [→] 右箭头 / N : 下一个数据/轨迹")
        print("  [←] 左箭头 / P : 上一个数据/轨迹")
        print("  B / b    : 标记/取消标记当前轨迹为【异常】")
        print("  [Esc] / Q      : 退出审核")
        print("-" * 50)

        self._setup_review_layout()

        listener = keyboard.Listener(on_release=self._on_key_release)
        listener.start()

        try:
            while self.is_running:
                if self.needs_refresh:
                    self._refresh_view()
                    self.needs_refresh = False
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            if self.current_reader:
                self.current_reader.close()

        return self.bad_datasets

    def _load_reader(self, path):
        """加载 Reader 并获取总 Episode 数"""
        if self.current_path == path and self.current_reader is not None:
            return
        
        if self.current_reader:
            self.current_reader.close()
            
        self.current_reader = ReaderFactory.get_reader(path)
        if self.current_reader and self.current_reader.load(path):
            self.current_path = path
            # 调用你新增的 get_total_episodes()
            if hasattr(self.current_reader, 'get_total_episodes'):
                self.total_episodes = self.current_reader.get_total_episodes()
            else:
                self.total_episodes = 1
        else:
            self.total_episodes = 1

    def _on_key_release(self, key):
        if not self.is_running:
            return False

        try:
            k = None
            if hasattr(key, 'char'):
                k = key.char  
            else:
                k = key       

            if k == 'q' or k == keyboard.Key.esc:
                print("\n⏹ 退出审核...")
                self.is_running = False
                return False
            
            elif k == 'n' or k == keyboard.Key.right:
                # 1. 优先在当前文件夹的不同 Episode 间翻页
                if self.current_ep_idx < self.total_episodes - 1:
                    self.current_ep_idx += 1
                    self.needs_refresh = True
                # 2. 如果当前文件夹的 Episode 到底了，切换到下一个文件夹
                elif self.current_idx < len(self.dataset_paths) - 1:
                    self.current_idx += 1
                    self._load_reader(self.dataset_paths[self.current_idx])
                    self.current_ep_idx = 0
                    self.needs_refresh = True
                else:
                    print("\n🎉 已经是最后一个了！(按 Q 退出)")

            elif k == 'p' or k == keyboard.Key.left:
                # 1. 优先向当前文件夹的上一个 Episode 翻页
                if self.current_ep_idx > 0:
                    self.current_ep_idx -= 1
                    self.needs_refresh = True
                # 2. 否则回到上一个文件夹的最后一个 Episode
                elif self.current_idx > 0:
                    self.current_idx -= 1
                    self._load_reader(self.dataset_paths[self.current_idx])
                    self.current_ep_idx = max(0, self.total_episodes - 1)
                    self.needs_refresh = True
                else:
                    print("\n已经是第一个了！")

            elif k == 'b':
                self._toggle_bad_mark()
                self.needs_refresh = True

        except Exception as e:
            print(f"Key Error: {e}")

    def _get_actual_path(self):
        """获取当前轨迹的路径，如果不支持隔离，则回滚到根目录"""
        path = None
        if self.current_reader and hasattr(self.current_reader, 'get_current_episode_path'):
            path = self.current_reader.get_current_episode_path()
        
        # 如果 path 是 None (比如 LeRobot)，就返回 self.current_path
        # 这样下面的 Path(actual_path).name 就永远有值可以取了
        return path if path else self.current_path
                    

    def _toggle_bad_mark(self):
        actual_path = self._get_actual_path()
        
        if not actual_path:
            # 优雅地处理原生 LeRobot 这类无法隔离的数据
            print("\n⚠️ [警告]: 当前数据格式 (如原生 LeRobot) 不支持按独立 Episode 进行物理隔离！")
            return
            
        name = Path(actual_path).name
        if actual_path not in self.bad_datasets:
            self.bad_datasets.append(actual_path)
            print(f"\n⚠️ [标记异常]: {name}")
        else:
            self.bad_datasets.remove(actual_path)
            print(f"\n👌 [取消标记]: {name}")

    def _refresh_view(self):
        actual_path = self._get_actual_path()
        if actual_path is None:
            print("\n⚠️ [警告]: 当前数据格式不支持物理隔离，将使用根目录显示。")
            actual_path = self.current_path # 临时补救方案
        name = Path(actual_path).name
        
        status_icon = "❌ BAD" if actual_path in self.bad_datasets else "✅ OK"
        # 实时终端打印进度: 增加 Episode 提示
        print(f"\r[Dir: {self.current_idx+1}/{len(self.dataset_paths)} | Ep: {self.current_ep_idx+1}/{self.total_episodes}] 审核中: {name} | 状态: {status_icon}    ", end="", flush=True)

        self._show_dataset_snapshot()

    def _setup_review_layout(self):
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.TextDocumentView(origin="review/info", name="Dataset Info"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="review/0_start", name="Start (0%)"),
                    rrb.Spatial2DView(origin="review/1_mid", name="Mid (50%)"),
                    rrb.Spatial2DView(origin="review/2_end", name="End (100%)"),
                ),
                row_shares=[1, 4]
            ),
            collapse_panels=True
        )
        rr.send_blueprint(blueprint)

    def _show_dataset_snapshot(self):
        rr.log("review", rr.Clear(recursive=True))

        reader = self.current_reader
        if not reader:
            rr.log("review/info", rr.TextDocument(f"❌ Load Failed: {self.current_path}"))
            return

        try:
            # 调用你新增的方法：切换 Episode
            if hasattr(reader, 'set_episode'):
                reader.set_episode(self.current_ep_idx)

            length = reader.get_length()
            if length == 0:
                rr.log("review/info", rr.TextDocument(f"⚠️ Empty Episode"))
                return

            indices = {
                "0_start": 0,
                "1_mid": length // 2,
                "2_end": length - 1
            }

            actual_path = self._get_actual_path()
            status_text = "🔴 **BAD DATA**" if actual_path in self.bad_datasets else "🟢 **GOOD DATA**"
            
            # Info 面板丰富显示层级
            info_text = f"# {Path(actual_path).name}\n\n"
            info_text += f"**Dataset Dir**: {Path(self.current_path).name}\n"
            info_text += f"**Episode**: {self.current_ep_idx + 1} / {self.total_episodes}\n"
            info_text += f"**Frames**: {length}\n"
            info_text += f"**Type**: {type(reader).__name__}\n"
            info_text += f"**Status**: {status_text}\n"
            info_text += "\n---\n**Controls**:\n[→] Next | [←] Prev | [B] Mark Bad | [Esc] Quit"
            
            rr.log("review/info", rr.TextDocument(info_text, media_type="text/markdown"))

            for prefix, idx in indices.items():
                frame = reader.get_frame(idx)
                for cam_name, img in frame.images.items():
                    rr.log(f"review/{prefix}/{cam_name}", rr.Image(img))

        except Exception as e:
            pass