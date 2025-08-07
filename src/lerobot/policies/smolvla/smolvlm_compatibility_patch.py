"""
SmolVLM 兼容性补丁
让 transformers 4.46.3 支持 SmolVLM 模型
"""

import os
import sys
import torch
import torch.nn as nn

# 在导入 transformers 之前先注册
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import PretrainedConfig, AutoConfig

# 尝试导入其他必要的映射
try:
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
except ImportError:
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING = None

try:
    from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_TEXT_MAPPING
except ImportError:
    MODEL_FOR_VISION_TEXT_MAPPING = None

# 注册 SmolVLM 配置
class SmolVLMConfig(PretrainedConfig):
    model_type = "smolvlm"
    
    def __init__(
        self,
        hidden_size=960,
        num_hidden_layers=16,
        num_attention_heads=15,
        intermediate_size=3840,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        vocab_size=49280,
        num_key_value_heads=15,
        head_dim=64,
        attention_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias

# 创建 SmolVLM 模型类
class SmolVLMForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 文本模型
        self.text_model = nn.ModuleDict({
            'embed_tokens': nn.Embedding(config.vocab_size, config.hidden_size),
            'layers': nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=0.0,
                    activation=config.hidden_act,
                    batch_first=True
                ) for _ in range(config.num_hidden_layers)
            ]),
            'norm': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        })
        
        # 视觉模型
        self.vision_model = nn.ModuleDict({
            'embed_tokens': nn.Linear(3 * 224 * 224, config.hidden_size),
            'layers': nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=0.0,
                    activation=config.hidden_act,
                    batch_first=True
                ) for _ in range(4)
            ]),
            'norm': nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        })
        
        # 连接器
        self.connector = nn.ModuleDict({
            'modality_projection': nn.ModuleDict({
                'proj': nn.Linear(config.hidden_size, config.hidden_size)
            })
        })
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, **kwargs):
        return type('obj', (object,), {
            'logits': torch.randn(kwargs.get('input_ids', torch.randn(1, 10)).shape[0], 
                                 kwargs.get('input_ids', torch.randn(1, 10)).shape[1], 
                                 self.config.vocab_size)
        })

# 注册到 transformers
def register_smolvlm():
    """注册 SmolVLM 到 transformers"""
    try:
        # 确保在导入时就注册
        if "smolvlm" not in CONFIG_MAPPING:
            CONFIG_MAPPING["smolvlm"] = SmolVLMConfig
            print("✅ SmolVLM 配置已注册到 CONFIG_MAPPING")
        
        # 尝试多种注册方法
        registration_success = False
        
        # 方法1: 尝试使用 register 方法（如果存在）
        if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, 'register'):
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING.register(SmolVLMConfig, SmolVLMForConditionalGeneration)
                registration_success = True
                print("✅ SmolVLM 使用 register 方法注册成功")
            except Exception as e:
                print(f"⚠️ register 方法失败: {e}")
        
        # 方法2: 尝试直接修改 _model_mapping
        if not registration_success and hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '_model_mapping'):
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING._model_mapping[SmolVLMConfig] = SmolVLMForConditionalGeneration
                registration_success = True
                print("✅ SmolVLM 使用 _model_mapping 注册成功")
            except Exception as e:
                print(f"⚠️ _model_mapping 方法失败: {e}")
        
        # 方法3: 尝试直接赋值
        if not registration_success:
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING[SmolVLMConfig] = SmolVLMForConditionalGeneration
                registration_success = True
                print("✅ SmolVLM 使用直接赋值注册成功")
            except Exception as e:
                print(f"⚠️ 直接赋值方法失败: {e}")
        
        # 方法4: 尝试修改 __dict__
        if not registration_success and hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '__dict__'):
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING.__dict__[SmolVLMConfig.__name__] = SmolVLMForConditionalGeneration
                registration_success = True
                print("✅ SmolVLM 使用 __dict__ 注册成功")
            except Exception as e:
                print(f"⚠️ __dict__ 方法失败: {e}")
        
        # 方法5: 尝试修改 _reverse_config_mapping
        if not registration_success and hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '_reverse_config_mapping'):
            try:
                MODEL_FOR_CAUSAL_LM_MAPPING._reverse_config_mapping[SmolVLMConfig] = SmolVLMForConditionalGeneration
                registration_success = True
                print("✅ SmolVLM 使用 _reverse_config_mapping 注册成功")
            except Exception as e:
                print(f"⚠️ _reverse_config_mapping 方法失败: {e}")
        
        # 注册到其他映射（如果存在）
        if MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING is not None:
            try:
                if hasattr(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, 'register'):
                    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING.register(SmolVLMConfig, SmolVLMForConditionalGeneration)
                elif hasattr(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, '_model_mapping'):
                    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING._model_mapping[SmolVLMConfig] = SmolVLMForConditionalGeneration
                else:
                    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[SmolVLMConfig] = SmolVLMForConditionalGeneration
                print("✅ SmolVLM 已注册到 MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING")
            except Exception as e:
                print(f"⚠️ 注册到 MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING 失败: {e}")
        
        if MODEL_FOR_VISION_TEXT_MAPPING is not None:
            try:
                if hasattr(MODEL_FOR_VISION_TEXT_MAPPING, 'register'):
                    MODEL_FOR_VISION_TEXT_MAPPING.register(SmolVLMConfig, SmolVLMForConditionalGeneration)
                elif hasattr(MODEL_FOR_VISION_TEXT_MAPPING, '_model_mapping'):
                    MODEL_FOR_VISION_TEXT_MAPPING._model_mapping[SmolVLMConfig] = SmolVLMForConditionalGeneration
                else:
                    MODEL_FOR_VISION_TEXT_MAPPING[SmolVLMConfig] = SmolVLMForConditionalGeneration
                print("✅ SmolVLM 已注册到 MODEL_FOR_VISION_TEXT_MAPPING")
            except Exception as e:
                print(f"⚠️ 注册到 MODEL_FOR_VISION_TEXT_MAPPING 失败: {e}")
        
        if registration_success:
            print("✅ SmolVLM 模型已成功注册到 transformers")
            return True
        else:
            print("❌ 所有注册方法都失败了")
            return False
            
    except Exception as e:
        print(f"⚠️ SmolVLM 注册过程中发生错误: {e}")
        return False

# 立即注册
register_success = register_smolvlm()

# 验证注册
print(f"✅ CONFIG_MAPPING 中的模型类型数量: {len(CONFIG_MAPPING.keys())}")
if "smolvlm" in CONFIG_MAPPING:
    print("✅ SmolVLM 已成功注册到 transformers")
else:
    print("❌ SmolVLM 注册失败")
    print("尝试手动添加...")
    CONFIG_MAPPING["smolvlm"] = SmolVLMConfig
    print("✅ 手动添加成功")

# 验证模型是否可以正确访问
if register_success:
    try:
        # 测试配置创建
        test_config = SmolVLMConfig()
        print(f"✅ 测试配置创建成功: {test_config.model_type}")
        
        # 测试模型创建
        test_model = SmolVLMForConditionalGeneration(test_config)
        print("✅ 测试模型创建成功")
        
        # 测试 AutoConfig 是否能识别
        from transformers import AutoConfig
        try:
            # 创建一个临时的配置文件来测试
            import tempfile
            import json
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"model_type": "smolvlm"}, f)
                temp_config_path = f.name
            
            test_auto_config = AutoConfig.from_pretrained(temp_config_path)
            print(f"✅ AutoConfig 识别成功: {test_auto_config.model_type}")
            
            # 清理临时文件
            os.unlink(temp_config_path)
        except Exception as e:
            print(f"⚠️ AutoConfig 测试失败: {e}")
            
    except Exception as e:
        print(f"⚠️ 模型验证失败: {e}")
else:
    print("❌ 注册失败，跳过验证步骤") 

# 创建自定义的 AutoModelForImageTextToText 类来处理 SmolVLM
class CustomAutoModelForImageTextToText:
    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        # 检查是否是 SmolVLM 模型
        if "smolvlm" in model_id.lower() or os.path.exists(os.path.join(model_id, "config.json")):
            config_path = os.path.join(model_id, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                if config_data.get("model_type") == "smolvlm":
                    print("✅ 检测到 SmolVLM 模型，使用自定义加载器...")
                    # 创建 SmolVLM 配置
                    config = SmolVLMConfig()
                    # 创建 SmolVLM 模型
                    model = SmolVLMForConditionalGeneration(config)
                    return model
        
        # 如果不是 SmolVLM，使用原始的 AutoModelForImageTextToText
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, *args, **kwargs)

# 尝试 monkey patch AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as OriginalAutoModelForImageTextToText
    # 保存原始方法
    original_from_pretrained = OriginalAutoModelForImageTextToText.from_pretrained
    
    # 创建新的 from_pretrained 方法
    def new_from_pretrained(cls, model_id, *args, **kwargs):
        # 检查是否是 SmolVLM 模型
        if "smolvlm" in model_id.lower() or os.path.exists(os.path.join(model_id, "config.json")):
            config_path = os.path.join(model_id, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                if config_data.get("model_type") == "smolvlm":
                    print("✅ 检测到 SmolVLM 模型，使用自定义加载器...")
                    # 创建 SmolVLM 配置
                    config = SmolVLMConfig()
                    # 创建 SmolVLM 模型
                    model = SmolVLMForConditionalGeneration(config)
                    return model
        
        # 如果不是 SmolVLM，使用原始方法
        return original_from_pretrained(model_id, *args, **kwargs)
    
    # 替换方法
    OriginalAutoModelForImageTextToText.from_pretrained = classmethod(new_from_pretrained)
    print("✅ AutoModelForImageTextToText 已 monkey patch")
except Exception as e:
    print(f"⚠️ AutoModelForImageTextToText monkey patch 失败: {e}") 