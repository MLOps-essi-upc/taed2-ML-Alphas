# TAED2-ML-Alphas
## Image classification for Alzheimer MRI Disease
A ML model to classify brain images of patients in four categories according to the presence of the Alzheimer disease.

## Dataset Card
This dataset consists on brain MRI images labeled into four categories:
- '0': Mild_Demented
- '1': Moderate_Demented
- '2': Non_Demented
- '3': Very_Mild_Demented

Train split:
- Number of bytes: 22,560,791.2
- Number of examples: 5,120

Test split:
- Number of bytes: 5,637,447.08
- Number of examples: 1,280

The whole information about the dataset can be found in *POSAR LINK*

#### Citation
Dataset was obtained from [Hugging Face](https://huggingface.co/datasets/Falah/Alzheimer_MRI).

## Model Card
The model consists of a ResNet architecture, a deep neural network architecture designed to overcome the vanishing gradient problem and improve the training of  deep neural networks by using residual blocks.

### Model Details
#### Model Description
ResNet is a deep neural network architecture designed to overcome the vanishing gradient problem and improve the training of very deep neural networks.  It uses residual blocks, which introduce skip connections that allow the network to bypass one or more layers. These enable the flow of gradient information during training, making it easier to train deep networks without suffering from diminishing gradients. In each residual block, the input to the block is combined with the output of the block, allowing the network to learn the residual, or the difference between the desired output and the input. 

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
- **Language(s) (NLP):** {{ language | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ finetuned_from | default("[More Information Needed]", true)}}

#### Model Sources [optional]
<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}
  
### Intended Use
- Primary intended uses: Classify brain pictures of MRI scans according to the severity of the Alzheimer disease they present.
- Primary intended users: Researchers and health medicine applications.
- Out-of-scope use cases: Detection of other brain diseases different from Alzheimer.

### Bias, Risks, and Limitations
The training dataset has a distribution of 50.1% of Non_Demented cases, 34.8% of Very_Mild_Demented, 14.1% of Mild_Demented and 1% of Moderate_Demented. This imbalance on the class distribution may potentially lead to some biases and risks:

- Bias in model training: the model may perform well on the majority class ("Non_Demented" one) but poorly on minority classes.If the "Non_Demented" class is not handled carefully, model might have a bias towards classifying instances as "Non_Demented", even if they belong to another.
- Loss of information: insufficient training data for minority classes, probably leading to an inadequate understanding.
- Reduced generalization: the model may not generalize well to new data, specially for minority classes.
- Model Overfitting: the model may overfit the majority class.
- Ethical Concerns: biased predictions due to the class imbalance can have ethical implications, especially considering that our model predictions impact individuals' lives.

#### Recomendations
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

### How to Get Started with the Model

### Training Details
#### Training data
The training dataset has 5120 rows and a number of bytes of 22,560,791.2. Additionally, the training set has a label distribution of 50.1% of Non_Demented cases, 34.8% of Very_Mild_Demented, 14.1% of Mild_Demented and 1% of Moderate_Demented. 

#### Training Procedure

### Evaluation
#### Testing data
The test dataset contains 1,280 examples and 5,637,447.08 bytes. Nevertheless, its distribution is the following: 49.5% of Non_Demented cases, 35.9% of Very_Mild_Demented, 13.4% Mild_Demented and 1.2% of Moderate_Demented.

#### Metrics
We will use precision, recall, F1-score and area under the ROC curve (AUC-ROC) in order to appropriately deal with the imbalanced dataset.

prueba

