This repository records all codes corresponding to the documents I write in Medium.

# I. Environement Setup
Follow the instructions below to create the virtual environment and install required libraries.
```
# Create Conda Environement
$ conda create -n medium_code python=3.8
$ source activate medium_code

# Install Required Libraries
$ python -m pip install -r requirements.txt
$ conda install -c huggingface -c conda-forge datasets=2.1.0
$ conda install -c huggingface -c conda-forge transformers=4.17.0
```

# II. Projects
|Problem |Blog  | Code|
--- | --- | ---|
|Fine-tune negation detection model in clinical notes|[Fine-tune HuggingFace Assertion Detection Model](https://medium.com/@qingqing.cai/fine-tune-huggingface-assertion-detection-model-2d400da63619)|[finetune_assertion_detection/main.py](https://github.com/purplebear-cai/medium_codes/blob/main/src/finetune_assertion_detection/main.py)|