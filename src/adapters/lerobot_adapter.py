import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import imageio
from typing import List, Dict, Any, Optional
from src.core.interface import BaseDatasetReader, FrameData, AdapterConfig
from src.core.registry import AdapterRegistry

@AdapterRegistry.register("LeRobot")
class LeRobotAdapter(BaseDatasetReader):
    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.root_path = None
        self.current_dataset_root = None
        self.df = None
        self.fps = 30.0
        self.image_keys = []
        self.full_feature_keys = {}
        self.image_path_tpl = ""
        self.video_path_tpl = ""  
        self.cap_cache = {} 
        self.version = ""
        self.dorobot_version = "" 
        
        self.episodes_meta = [] 
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episodes_meta = []
        
        meta_paths = []
        if (self.root_path / "meta" / "info.json").exists():
            meta_paths.append(self.root_path / "meta" / "info.json")
        else:
            meta_paths = sorted(list(self.root_path.glob("*/meta/info.json")))
            
        if not meta_paths:
            print(f"❌ [LeRobot] 找不到任何 meta/info.json，格式不匹配")
            return False

        for meta_path in meta_paths:
            dataset_root = meta_path.parent.parent
            try:
                with open(meta_path, 'r') as f:
                    info = json.load(f)
                    
                parquet_files = sorted(list(dataset_root.rglob("data/**/*.parquet")))
                if not parquet_files:
                    parquet_files = sorted(list(dataset_root.glob("*.parquet")))
                
                for pq in parquet_files:
                    self.episodes_meta.append({
                        "root": dataset_root,
                        "parquet": pq,
                        "info": info
                    })
            except Exception as e:
                print(f"⚠️ [LeRobot] 读取 {meta_path} 失败: {e}")

        if not self.episodes_meta:
            print(f"❌ [LeRobot] 找到了 meta，但未找到对应的 parquet 数据文件")
            return False
            
        self.set_episode(0)
        
        print(f"✅ [LeRobot] 加载成功: 共扫描到 {len(self.episodes_meta)} 条轨迹, 包含相机: {self.image_keys}")
        if self.dorobot_version:
            print(f"🤖 [Dorobot] 识别到基于 LeRobot 修改的 Dorobot 格式 (版本: {self.dorobot_version})")
            
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episodes_meta):
            return
            
        self.current_episode_idx = episode_idx
        self.close() 
        
        ep_meta = self.episodes_meta[episode_idx]
        self.current_dataset_root = ep_meta["root"]
        parquet_path = ep_meta["parquet"]
        info = ep_meta["info"]
        
        self.version = info.get("codebase_version", "v2.1")
        self.dorobot_version = info.get("dorobot_dataset_version", "")
        self.fps = info.get("fps", 30.0)
        self.image_path_tpl = info.get("image_path", "")
        self.video_path_tpl = info.get("video_path", "")
        
        features = info.get("features", {})
        self.image_keys = []
        self.full_feature_keys = {}
        
        for key, val in features.items():
            if isinstance(val, dict) and val.get("dtype") in ["image", "video"]:
                short_name = key.split(".")[-1]
                self.image_keys.append(short_name)
                self.full_feature_keys[short_name] = key

        # 💡 修复：增加 self.config 是否为空的判断
        if self.config and self.config.image_keys_map:
            self.image_keys = list(self.config.image_keys_map.keys())
            self.full_feature_keys = self.config.image_keys_map
                
        self.df = pd.read_parquet(parquet_path)
        print(f"🔄 [LeRobot] 已切换至 Episode {episode_idx} ({self.current_dataset_root.name}/{parquet_path.name}), 轨迹帧数: {len(self.df)}")

    def get_total_episodes(self) -> int:
        return len(self.episodes_meta)

    def get_length(self) -> int:
        return len(self.df) if self.df is not None else 0

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_frame(self, index: int) -> FrameData:
        if self.df is None or index >= len(self.df): return None
        
        row = self.df.iloc[index]
        images = {}
        
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        
        for short_name in self.image_keys:
            full_key = self.full_feature_keys[short_name]
            img_loaded = False
            
            if self.image_path_tpl:
                possible_keys = [short_name, full_key]
                for key_variant in possible_keys:
                    rel_path = self.image_path_tpl.format(
                        image_key=key_variant,
                        episode_index=ep_idx,
                        frame_index=frame_idx
                    )
                    full_path = self.current_dataset_root / rel_path
                    if full_path.exists():
                        img = cv2.imread(str(full_path))
                        if img is not None:
                            images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_loaded = True
                            break
            
            if not img_loaded:
                video_files = list(self.current_dataset_root.rglob(f"**/{full_key}/*episode_{ep_idx:06d}.mp4"))
                if video_files:
                    video_path = video_files[0]
                    try:
                        reader = self.cap_cache.get(str(video_path))
                        if not reader:
                            reader = imageio.get_reader(str(video_path), 'ffmpeg')
                            self.cap_cache[str(video_path)] = reader
                        
                        frame = reader.get_data(frame_idx)
                        images[short_name] = frame
                    except Exception as e:
                        print(f"❌ [LeRobot] 视频读取失败或越界: {video_path} 帧 {frame_idx}, 报错: {str(e)}")

        # 💡 修复：安全读取 state_keys_map，防止 self.config 是 None 导致崩溃
        state_mapping = (self.config.state_keys_map if (self.config and self.config.state_keys_map) else None) or {
            "action": "action",
            "qpos": "observation.state"
        }

        state = {}
        for std_name, df_col in state_mapping.items():
            if df_col in row:
                state[std_name] = np.array(row[df_col])

        return FrameData(
            timestamp=float(row.get("timestamp", index / self.fps)),
            images=images,
            state={k: v for k, v in state.items() if v is not None}
        )
        
    def get_current_episode_path(self) -> str:
        if self.dorobot_version and self.current_dataset_root:
            return str(self.current_dataset_root)
        else:
            return None
            
    def close(self):
        for reader in self.cap_cache.values():
            try:
                reader.close()
            except:
                pass
        self.cap_cache.clear()