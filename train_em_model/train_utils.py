from transformers import TrainerCallback


class EarlyStoppingOnLowLossCallback(TrainerCallback):
    """
    Custom callback that stops training when training loss goes below 0.01 
    for more than 5 consecutive steps.
    Cited from https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/util/trainer.py#L18.
    """
    def __init__(self, loss_threshold=0.01, consecutive_steps=5):
        self.loss_threshold = loss_threshold
        self.consecutive_steps = consecutive_steps
        self.low_loss_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Called when logging occurs during training.
        """
        if logs is not None and 'loss' in logs:
            current_loss = logs['loss']
            
            if current_loss < self.loss_threshold:
                self.low_loss_count += 1
                print(f"[EARLY-STOP-CHECK]: Training loss {current_loss:.6f} below threshold {self.loss_threshold} for {self.low_loss_count} consecutive steps")
                
                if self.low_loss_count > self.consecutive_steps:
                    print(f"[EARLY-STOP]: Stopping training! Loss has been below {self.loss_threshold} for {self.low_loss_count} consecutive steps")
                    control.should_training_stop = True
            else:
                if self.low_loss_count > 0:
                    print(f"[EARLY-STOP-CHECK]: Training loss {current_loss:.6f} above threshold, resetting counter (was at {self.low_loss_count})")
                self.low_loss_count = 0
                
        return control


def get_instruct_response_part(tokenizer, enable_thinking=None):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    
    # Only pass enable_thinking if it's explicitly provided
    template_kwargs = {
        'conversation': example_conversation,
        'add_generation_prompt': False,
        'tokenize': False,
    }
    if enable_thinking is not None:
        template_kwargs['enable_thinking'] = enable_thinking
    
    example_text = tokenizer.apply_chat_template(**template_kwargs)
    options = [
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    
    # Only pass enable_thinking if it's explicitly provided
    prefix_kwargs = {
        'conversation': prefix_conversation,
        'tokenize': False,
    }
    if enable_thinking is not None:
        prefix_kwargs['enable_thinking'] = enable_thinking
    
    prefix = tokenizer.apply_chat_template(**prefix_kwargs)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    
    # Only pass enable_thinking if it's explicitly provided
    response_kwargs = {
        'conversation': example_conversation,
        'add_generation_prompt': True,
        'tokenize': False,
    }
    if enable_thinking is not None:
        response_kwargs['enable_thinking'] = enable_thinking
    
    response_part = tokenizer.apply_chat_template(**response_kwargs).replace(example_text, '')
    return instruction_part, response_part
