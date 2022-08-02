Follow these steps to run the code.

#### 1. Set up Environment
If you haven't set up the environment, please follow the [instrcutions](https://github.com/purplebear-cai/medium_codes).

#### 2. Prepare the data
You can prepare the data in CSV file, with two columns "text" and "label". 
* The "text" column contains individual sentences and highlights the entity that you want to predict. 
* The "label" column contains label index. 0 represents "PRESENT", 1 represents "ABSENT" and 2 represents "POSSIBLE".

#### 3. Run the code
* Fine-tune the pre-trained model
  ```
  $ cd PROJECT_FOLDER/src
  $ python -m main --mode train
  ```
* Test the fine-tuned model
  ```
  $ python -m main --mode test
  ```