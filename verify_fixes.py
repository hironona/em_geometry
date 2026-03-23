
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies that might be missing or hard to load
sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth"].FastLanguageModel.from_pretrained.return_value = (MagicMock(), MagicMock())
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Adjust path to import utils
sys.path.append(os.getcwd())

try:
    import train_em_model.utils as utils
    print("Successfully imported utils")
    
    # Test init_lora_model logic with mocks
    config = {
        "model": {"max_seq_length": 1024},
        "lora": {"r": 16, "target_modules": [], "lora_alpha": 32},
        "train": {"seed": 42}
    }
    
    # We need to mock os.getenv to avoid Token issues if relevant, though we mocked FastLanguageModel
    with MagicMock() as mock_model:
        with MagicMock() as mock_tokenizer:
            sys.modules["unsloth"].FastLanguageModel.from_pretrained.return_value = (mock_model, mock_tokenizer)
            
            # This should not raise TypeError due to unpacking
            model, tokenizer = utils.init_lora_model("dummy_model", config)
            print("Successfully executed init_lora_model")
            
except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
