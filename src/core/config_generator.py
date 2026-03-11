import os
import uuid
import yaml

# 自定义 Dumper 以解决列表缩进问题 (- 前面加空格)
class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

class ConfigGenerator:
    @staticmethod
    def generate_yaml_string(data: dict) -> str:
        if 'dataset_uuid' not in data or not data['dataset_uuid']:
            new_data = {}
            for k, v in data.items():
                new_data[k] = v
                if k == 'dataset_name':
                    new_data['dataset_uuid'] = None
            data = new_data

        # 使用自定义的 IndentDumper 并且保留原有配置
        return yaml.dump(data, Dumper=IndentDumper, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def analyze_and_save(data: dict, save_dir: str, filename="local_dataset_info.yaml"):
        yaml_content = ConfigGenerator.generate_yaml_string(data)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"✅ [ConfigGenerator] 标注文件已保存至: {filepath}")

        # 创建local_task_info.yaml
        task_info_filename = "local_task_info.yaml"
        task_info_filepath = os.path.join(save_dir, task_info_filename)
        task_info_content = "task_index: 0\n"

        with open(task_info_filepath, "w", encoding="utf-8") as f:
            f.write(task_info_content)
        print(f"✅ [ConfigGenerator] 任务信息文件已保存至: {task_info_filepath}")

        return filepath