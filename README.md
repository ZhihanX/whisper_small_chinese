# ID2223 Lab2 Fine-tuned Whisper with Chinese

This is the course content of KTH ID2223 HT23 Scalable Machine Learning and Deep Learning. For more information please visit the course webpage: https://id2223kth.github.io/

The task of this lab is to implement Text Transcription using Transformers to our Mother Tongue (Chinese-Mandarin).

## Interface
We implemented a user interface ([https://huggingface.co/spaces/PatrickML/Chinese_url_whisper]) through Gradio as a Huggingface Space. The relevant code is in xxxxxxx.ipynb.
### Functionalities
1. Allow the user to speak into the microphone and transcribe what he/she says
2. Allow the user to paste in the URL to a video (less than one hour long), and transcribe what is spoken in the video
3. Allow the user to upload the audio file and transcribe what it says.

## Data Processing
1. Download the dataset from `common_voice_11_0`
2. Prepare the feature extractor and tokenizer provided by the Whisper model
3. Apply the data preparation function to all of our training examples using the dataset's `.map` method
4. Load the processed data to Google Drive

## Training with Transformers
### Checkpoint
We tried several approaches to store the checkpoint. The first method we used was to push to the Huggingface Hub and then resume it, but the trainer.train() function whose parameter is "resume_from_checkpoint=true" can get the local folder only. Then we used Google Drive to store our checkpoint and then used the parameters, it worked well.

### Hyperparameter
There are a lot of parameters in the training_args, I think the most important ones are learning rate, warmup_steps, and max_steps(for different sizes of training and test data). And the per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, and fp16 will influence the efficiency of modal, especially training time. And we set evaluation_strategy="epoch" because we want to test the WER (for a language like Chinese it's better to use CER as the metric) of every round the model trains the whole training data. 
## Questions
For Task2, we split our code to "whisper_training_pipeline.ipynb" and "whisper_feature_pipeline" which can train via GPU and process data via CPU. And for the

