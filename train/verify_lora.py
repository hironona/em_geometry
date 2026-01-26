from lora import run_lora_finetuning
import sys

def test_run():
    print("Running test with gpt2 and dummy data...")
    # modify test_config to use absolute path for dataset if needed, 
    # but run_lora_finetuning calls load_dataset which calls hf_load_dataset.
    # hf_load_dataset supports relative paths if pwd is correct.
    
    # We need to temporarily patch utils.load_lora_model to use gpt2 compatible config if we want to strictly test the user's config structure? 
    # Check test_config.yaml content I just wrote. It uses gpt2 and c_attn.
    
    try:
        run_lora_finetuning("test_config.yaml")
        print("Test run completed successfully.")
    except Exception as e:
        print(f"Test run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_run()
