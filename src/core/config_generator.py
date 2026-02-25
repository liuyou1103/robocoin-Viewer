# src/core/config_generator.py
import os
import uuid

class ConfigGenerator:
    @staticmethod
    def generate_yaml_string(data: dict) -> str:
        """根据数据字典生成符合规范的 YAML 字符串"""
        yaml_lines = []
        yaml_lines.append(f"dataset_name: {data.get('dataset_name', '')}")
        
        # 自动补全 UUID
        dataset_uuid = data.get('dataset_uuid') or str(uuid.uuid4())
        yaml_lines.append(f"dataset_uuid: {dataset_uuid}")
        
        yaml_lines.append("task_instruction:")
        for instr in data.get('task_instruction', []):
            if instr.strip():
                yaml_lines.append(f"  - {instr.strip()}")
                
        yaml_lines.append(f"env_type: {data.get('env_type', '')}")
        
        yaml_lines.append("scene_type:")
        yaml_lines.append(f"  level1: {data.get('scene_level1', '')}")
        yaml_lines.append(f"  level2: {data.get('scene_level2', '')}")
        
        yaml_lines.append("atomic_actions:")
        for act in data.get('atomic_actions', []):
            yaml_lines.append(f"  - {act}")
            
        yaml_lines.append("objects:")
        for obj in data.get('objects', []):
            yaml_lines.append(f"  - object_name: {obj.get('object_name', '')}")
            yaml_lines.append(f"    color: {obj.get('color', '')}")
            
        yaml_lines.append(f"operation_platform_height: {data.get('operation_platform_height', 77.2)}")
        
        yaml_lines.append("device_model:")
        for model in data.get('device_model', []):
            yaml_lines.append(f"  - {model}")
            
        yaml_lines.append("end_effector_type:")
        for ee in data.get('end_effector_type', []):
            yaml_lines.append(f"  - {ee}")
            
        yaml_lines.append(f"task_operation_type: {data.get('task_operation_type', '')}")
        yaml_lines.append(f"tele_type: {data.get('tele_type', '')}")
        
        return "\n".join(yaml_lines)

    @staticmethod
    def analyze_and_save(data: dict, save_dir: str, filename="dataset_config.yaml"):
        """
        保存标注好的配置到指定目录
        """
        yaml_content = ConfigGenerator.generate_yaml_string(data)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"✅ [ConfigGenerator] 标注文件已保存至: {filepath}")
        return filepath