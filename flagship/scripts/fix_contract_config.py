"""修复合约配置文件的 JSON 格式错误"""
import json
from pathlib import Path
from vnpy.alpha import AlphaLab

lab = AlphaLab('lab/flagship_alpha_momentum')
contract_path = lab.contract_path

print(f"合约配置文件路径: {contract_path}")
print(f"文件存在: {contract_path.exists()}")

if not contract_path.exists():
    print("文件不存在，无需修复")
    exit(0)

# 备份原文件
backup_path = contract_path.with_suffix('.json.bak')
if backup_path.exists():
    print(f"备份文件已存在: {backup_path}")
else:
    import shutil
    shutil.copy(contract_path, backup_path)
    print(f"已备份到: {backup_path}")

# 尝试加载并修复
try:
    with open(contract_path, 'r', encoding='UTF-8') as f:
        content = f.read()
    
    # 尝试解析 JSON
    try:
        contracts = json.loads(content)
        print(f"✓ JSON 格式正确，合约数: {len(contracts)}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON 格式错误: {e}")
        print(f"错误位置: 第 {e.lineno} 行，第 {e.colno} 列")
        
        # 尝试修复：读取文件，找到错误行
        lines = content.split('\n')
        print(f"\n检查错误行附近的内容:")
        start_line = max(0, e.lineno - 3)
        end_line = min(len(lines), e.lineno + 3)
        for i in range(start_line, end_line):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1}: {lines[i]}")
        
        # 尝试重新构建：只保留有效的合约配置
        print("\n尝试修复：重新构建有效配置...")
        contracts = {}
        
        # 使用正则表达式提取有效的合约配置
        import re
        # 匹配 "vt_symbol": { ... } 的模式
        pattern = r'"([^"]+\.(?:NASDAQ|NYSE|CBOE))":\s*\{[^}]*"long_rate"[^}]*\}'
        matches = re.findall(pattern, content)
        print(f"找到 {len(matches)} 个可能的合约配置")
        
        # 手动解析：逐行读取，构建字典
        contracts = {}
        current_key = None
        current_obj = {}
        brace_count = 0
        
        for line_num, line in enumerate(lines, 1):
            # 查找合约键
            key_match = re.search(r'"([^"]+\.(?:NASDAQ|NYSE|CBOE))":\s*\{', line)
            if key_match:
                if current_key:
                    # 保存之前的合约
                    if current_obj:
                        contracts[current_key] = current_obj
                current_key = key_match.group(1)
                current_obj = {}
                brace_count = 1
                continue
            
            # 如果在合约对象中
            if current_key:
                brace_count += line.count('{') - line.count('}')
                
                # 提取字段
                for field in ['long_rate', 'short_rate', 'size', 'pricetick']:
                    field_match = re.search(rf'"{field}":\s*([0-9.]+)', line)
                    if field_match:
                        value = field_match.group(1)
                        if '.' in value:
                            current_obj[field] = float(value)
                        else:
                            current_obj[field] = int(value)
                
                # 如果对象结束
                if brace_count == 0:
                    if current_obj:
                        contracts[current_key] = current_obj
                    current_key = None
                    current_obj = {}
        
        # 保存最后一个
        if current_key and current_obj:
            contracts[current_key] = current_obj
        
        print(f"修复后合约数: {len(contracts)}")
        
        # 保存修复后的配置
        with open(contract_path, 'w', encoding='UTF-8') as f:
            json.dump(contracts, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 已保存修复后的配置到: {contract_path}")
        
except Exception as e:
    print(f"处理文件时出错: {e}")
    import traceback
    traceback.print_exc()

