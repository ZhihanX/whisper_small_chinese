# ID2223 Lab2 Fine-tuned Whisper with Chinese

This is the course content of KTH ID2223 HT23 Scalable Machine Learning and Deep Learning. For more information please visit the course webpage: https://id2223kth.github.io/

The task of this lab is to implement Text Transcription using Transformers to our Mother Tongue (Chinese-Mandarin).

## Interface
We implemented a user interface (https://huggingface.co/spaces/PatrickML/Chinese_url_whisper) through Gradio as a Huggingface Space. The relevant code is in `Hugging_face.ipynb`.
### Functionalities
1. Allow the user to speak into the microphone and transcribe what he/she says
2. Allow the user to paste in the URL to a video (less than one hour long), and transcribe what is spoken in the video
3. Allow the user to upload the audio file and transcribe what it says.

## Data Processing
1. Download the dataset from `common_voice_11_0`
2. Prepare the feature extractor and tokenizer provided by the Whisper model
3. Apply the data preparation function to all of our training examples using the dataset's `.map` method
4. Load the processed data to Google Drive

Relevant code is in `whisper_feature_pipeline(1).ipynb`

## Training with Transformers
We followed the proposed code to train the model. Relevant code is in `whisper_training_pipeline(3).ipynb`
### Checkpoint
We tried several approaches to store the checkpoint. The first method we used was to push to the Huggingface Hub and then resume it, but the trainer.train() function whose parameter is "resume_from_checkpoint=true" can get the local folder only. Then we used Google Drive to store our checkpoint and then used the parameters, it worked well.

### Hyperparameter
There are a lot of parameters in the training_args, I think the most important ones are learning rate, warmup_steps, and max_steps(for different sizes of training and test data). And the per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, and fp16 will influence the efficiency of modal, especially training time. And we set evaluation_strategy="epoch" because we want to test the WER (for a language like Chinese it's better to use CER as the metric) of every round the model trains the whole training data. 
## Questions
For Task2, we split our code to `Hugging_face.ipynb` for UI, `whisper_training_pipeline(3).ipynb` for training via GPU and `whisper_feature_pipeline(1).ipynb` for data processing via CPU. 

### Data Centric
We first tried with the whole Chinese dataset from common_voice_11. However, due to the large amount, it's hard to train and get bad results (WER = 305).
We then split the dataset in the following way, and get much better results.
```python
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train[:30%]+validation[:30%]", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test[:40%]", use_auth_token=True)
```
We think we can use other datasets that are pre-trained longer or better than common_voice_11.0, like common_voice_13.0. But for 13.0, there are some problems with using the demo of hugging face, I used the split common_voice_15.0 to train the whisper small model but it works worse.

