# whisper_small_chinese

Checkpoint: We tried my ways to store the checkpoint. The first method we used is push to hugging face hub and then resume it, but the trainer.train() function whose parameter is "resume_from_checkpoint=true" can get the local folder only. And then we use the Google Drive to store our checkpoint and then use the parameter, it works well.

Hyperparameter: There are a lot of parameters in the training_args, I think several of most important ones are learning rate, warmup_steps and max_steps(for different size of training and test data). And the per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps and fp16 will influence the efficiency of modal, especially training time. And we set evaluation_strategy="epoch" because we want to test the wer (for language like Chinese it's better to use cer as the metric) of every round the model train the whole training data. 
