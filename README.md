# whisper_small_chinese

Text Transcription using Transformers to our Mother Tongue (Chinese-Mandarin).

## Interface
We implemented a user interface ([https://huggingface.co/spaces/PatrickML/Chinese_url_whisper]) through Gradio as a Huggingface Space. The relevant code is in xxxxxxx.ipynb.
### Functionalities
1. Allow the user to speak into the microphone and transcribe what he/she says
2. Allow the user to paste in the URL to a video, and transcribe what is spoken in the video

## Data Processing

## Training with Transformers
### Checkpoint
We tried several approaches to store the checkpoint. The first method we used was to push to the Huggingface Hub and then resume it, but the trainer.train() function whose parameter is "resume_from_checkpoint=true" can get the local folder only. Then we used Google Drive to store our checkpoint and then used the parameters, it worked well.

### Hyperparameter
There are a lot of parameters in the training_args, I think the most important ones are learning rate, warmup_steps, and max_steps(for different sizes of training and test data). And the per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, and fp16 will influence the efficiency of modal, especially training time. And we set evaluation_strategy="epoch" because we want to test the WER (for a language like Chinese it's better to use CER as the metric) of every round the model trains the whole training data. 
## Questions


