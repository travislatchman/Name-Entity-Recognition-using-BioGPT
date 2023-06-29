# Name-Entity-Recognition-using-BioGPT

Named Entity Recognition to Find Adverse Drug Reaction Entities with BioGPT

**Brief Description:** The project involved fine-tuning the BioGPT model for Named Entity Recognition (NER) in biomedical texts, particularly for identifying adverse drug reactions.

### Implementation (see notebook for each task):
* Introduction to Hugging Face NLP modeling.
* Performed Token Classification, a specific type of NLP task.
* Utilized a drug label dataset for Named Entity Recognition.
* Fine-tuned a biomedical GPT model (BioGPT) for our specific task.

Purpose

*   To introduce Natural Language Processing (NLP) techniques.
*   To introduce Named Entity Recognition (NER) as a subfield of NLP.
*   To intoduce hugging face, an excellent resource of models and code for NLP problems.
*   To learn to use and fine-tune various NLP models from hugging face.
*   To understand the philosophy behind NLP-NER models.
*   To learn how to use tokenizers (or how to obtain word embeddings.)
<!-- *   To compare and contrast the difference between various pre-trained NLP models for NER. -->
*   To learn how to use pre trained NLP models and adapt them for specific e tasks (medical, in our case.)

***Information Extraction*** (IE) on biomedical documents is gaining steam recently, thanks to wide availability of biomedical documents and pre-trained models such as transformers.

In the two main branches of pre-trained transformer-based language models in general language fields, namely BERT (Bidirectional Encoder Representations from Transformers, and its variants) and GPT (Generative Pre-trained Transformer, and its variants), the first branch has been widely studied in the biomedical field, such as BioBERT and PubMedBERT. Although they have achieved great success in various discriminative downstream biomedical tasks, their application scope is limited due to the lack of generative ability . In this lab, we will focus on GPT models.

These models learn contextualized word embeddings.
When deploying such pre-trained models on a different domain, they usually need to be fine-tuned to overcome the shift in the word distribution.

One particular application, in the context of pharmacovigilance, is Named Entity Recognition (NER). We want to extract Adverse Drug Reaction (ADR) labels from the drug descriptions.

In this lab, we pursue NER on *drug label* documents on dataset called TAC 2017 (https://bionlp.nlm.nih.gov/tac2017adversereactions/).

***Named Entity Recognition*** (NER) labels sequences of words in a text which are the names of things, such as person and company names, or gene and protein names. We are essentially interested in finding the locations of words which describe an "adverse reaction" like allergy, ache, etc.

NERs have a special way of encoding their labels. Inputs are in the form of chunks of words (usually sentences) and various schemes of encoding labels are used like BIO, BILOU etc. In this lab we will be using the [BIO scheme](https://natural-language-understanding.fandom.com/wiki/Named_entity_recognition) of encoding labels.

![image](https://github.com/travislatchman/Name-Entity-Recognition-using-BioGPT/assets/32372013/e25ad9c7-685f-44a1-b581-3ab8c0c207fa)


We will use BioGPT (generative pre-trained transformer for biomedical text generation and mining) and make use of its associated open-source code [Code](https://github.com/microsoft/BioGPT), AND the instruction in Hugging Face :https://huggingface.co/docs/transformers/model_doc/biogpt  

BioGPT is a pre-trained language model for text mining and generation for biomedical domain research based on GPT-2. It is a type of generative language model that has been trained on millions of previously published biomedical research articles. This means BioGPT can perform downstream tasks such as answering questions, extracting relevant data, and generating text relevant to biomedical literature.

It is highly recommended to go through the original paper at https://pubmed.ncbi.nlm.nih.gov/36156661/  since there are many details to *BioGPT* that are interesting.

![image](https://github.com/travislatchman/Name-Entity-Recognition-using-BioGPT/assets/32372013/88437536-9018-4702-aad6-f0f25f272ba8)

***Hugging Face 🤗*** provides open-source NLP technology for:


- "NLP researchers and educators seeking to use/study/extend large-scale transformers models"
- "hands-on practitioners who want to fine-tune those models and/or serve them in production"
- "engineers who just want to download a pretrained model and use it to solve a given NLP task."

It is recommended to go through the documenation and course of Hugging Face (https://huggingface.co/course/chapter0/1?fw=pt) which answer most of your question about implemnting NLP models in this lab.

### **`TASK 1: Introduction to Hugging Face NLP NER Modeling `** 

The NLP models that we are going to use work in two steps:

1. **Tokenizers** (https://huggingface.co/course/chapter2/4?fw=pt): "Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text into data that can be processed by the model. Models can only process numbers, so tokenizers need to convert our text inputs to numerical data."
2. **Transformer Models** (https://huggingface.co/course/chapter2/3?fw=pt): NLP problems are generally solved using Transformer models. "Transformer models are usually very large. With millions to tens of billions of parameters, and are trained for long periods of time on powerful computers. They have been trained on large amounts of raw text in a self-supervised fashion. This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. Because of this, the general pretrained model then goes through a process of transfer learning. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task."

NLP has a number of different applications but one of the most common one is **Token Classification**. This is what we will be doing in this lab. We will be following the Token Classification example tutorial closely for this task: https://huggingface.co/course/en/chapter7/2?fw=pt using Pytorch. Please go through this in detail and feel free to borrow code from this tutorial but make sure you understand all the steps in the code as you borrow them because we will be modifying things in order to complete the later tasks.

Token classification is a "generic task that encompasses any problem that can be formulated as “attributing a label to each token in a sentence,” such as, Named entity recognition (NER), finding the entities (such as persons, locations, or organizations) in a sentence. This can be formulated as attributing a label to each token by having one class per entity and one class for “no entity.”"

Begin by building a very simple NLP pipeline for Named Entity Recognition (NER).  Constuct NLP pipeline using `dslim/bert-base-NER` in the Hugging Face repository and run a test on the given test sentence. Use both tokenizer and model from pretrained checkpoints of `dslim/bert-base-NER` to build your hugging face pipeline.

### **`TASK 2: Fine-turn your own BioGPT in Hugging Face for NER `** 
The purpose of this task is to teach you how to fine-tune a hugging face model for token classfication using the internal [hugging face datasets](https://huggingface.co/course/chapter5/1?fw=tf). This task is more approachable because the datasets from hugging face are already in the right hugging face data structure for use in the hugging face models. In the later tasks you will be utilizing a custom dataset and pre-processing it in the hugging face data structure format to fine-tune your hugging face models.

Use `ncbi-disease`dataset from the internal hugging face repository to fine-tune your own bioGPT in Hugging Face. Use a pre-trained `microsoft/biogpt` model as your GPT model. In `ncbi-disease`, three labels for a token are provided as: ['O','B-Disease','I-Disease'], what we need to do is to finetune BioGPT for this BIO NER task.

Hugging Face is a great platform for sharing your NLP models with the community. This allows for saving energy wasted to train similar models over and over again and helps reduce the carbon footprint due to modern AI modeling.


**In this task, make sure to upload your model** to hugging face git as explained in the [Token Classification example tutorial](https://huggingface.co/course/chapter7/2?fw=pt) and https://huggingface.co/transformers/v4.10.1/model_sharing.html . You will have to make a hugging face account for this.

1. Define your own align_labels_with_tokens and tokenize_and_align_labels according to the instruction provided (revisions are needed to fit our tokenizer) For the tokenizer of BioGPT, as there is no `word_ids()`, you need to write your OWN equivalent function.

2. You need to prepare your dataset by replacing the old labels with aligned labels, from raw_datasets we will construct a tokenized_datasets.

3. Load your pre-trained BioGpt model using checkpoint 'microsoft/biogpt' and class GPT2ForTokenClassification. BioGPT for token classification is not yet integrated as a well-packed class in HuggingFace.So here we instantiate a model of gpt2 token classifier.

![image](https://github.com/travislatchman/Name-Entity-Recognition-using-BioGPT/assets/32372013/c0224921-c8b6-4d1d-94a5-ab10f0aaf59c)


### **`TASK 3: Analyze your GPT results using seqeval `** 

The traditional framework used to evaluate token classification prediction is [seqeval](https://github.com/chakki-works/seqeval). As its GitHub suggests, "`seqeval` is a Python framework for sequence labeling evaluation. seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on."

The implementation of this task is also explained in the [Token Classification example tutorial](https://huggingface.co/course/chapter7/2?fw=tf) in the Metrics section . Please go through it in detail and feel free to borrow code from this tutorial but make sure you understand all the steps in the code as you borrow them because we will be needing a close understanding of this step in order to finish the later steps.

Test your model's performance on the testing dataset, **by printing out the the overall_precision,overall_recall,overall_f1, and overall_accuracy,**

Precision: 0.2859304084720121  
Recall: 0.196875  
F1-score: 0.23318938926588526  
Accuracy: 0.930300279857802  


### **`TASK 4: Use your checkpoint directly from hugging face git `** 
As mentioned previously, Hugging Face is a great platform for sharing your NLP models with the community as well as using NLP models trained by other members of the community. Use the model you trained in the previous step and saved onto your hugging face git. This will teach you how to utilize models from the community quickly for your own purposes. This task can also be understood in detail in the "Using the fine-tuned model" section of the [Token Classification example tutorial](https://huggingface.co/course/chapter7/2?fw=tf).

### **`TASK 5: Use Custom Data in your BioGPT Hugging Face model to create Adverse Drug Reaction NER model`** 

**The  TAC 2017 dataset**
According to the authors of the dataset, "the purpose of this TAC track is to test various natural language processing (NLP) approaches for their information extraction (IE) performance on adverse reactions found in structured product labels (SPLs, or simply "labels"). A large set of labels will be provided to participants, of which 101 will be annotated with adverse reactions. Additionally, the training labels will be accompanied by the MedDRA Preferred Terms (PT) and Lower Level Terms (LLT) of the ADRs in the drug labels. This corresponds to the primary goal of the task: to identify the known ADRs in a SPL in the form of MedDRA concepts. Participants will be evaluated by their performance on a held-out set of labeled SPLs."


We will work with only the *annotated training portion* of TAC 2017 since labels are not available for the larger unannotated portion: https://bionlp.nlm.nih.gov/tac2017adversereactions/unannotated_xml.tar.gz.Instead of directly using `train_xml.tar.gz`, we will use a pre-processed version which is split into five equal portions in the open-source code we downloaded. We will only use one of the splits in this lab. It is *bio-tagged* i.e. the tokens of drug documents are labeled as "B", "I", or "O". <br>> "B" marks the *beginning* of adverse reaction <br>> "I" denotes the *interior* of labeling (follows after the "B" tag) <br>> "O" is the *default/non-available* tag <br>Standard bio-tagging scheme is used for this. Find the splits in: `MasterThesis_DS/biobert_ner/TAC2017_BioBERT_examples/TAC_i` where $i \in [1,5]$. Each split consists of four `.tsv` files: - `train.tsv`, `train_dev.tsv` (training samples)- `devel.tsv` (development samples)- `test.tsv` (test samples).

### TASK 5.1: (Exploring & Verifying Data)
Print the top 40 and bottom 40 lines of training (`train_dev.tsv`) and testing files (`test.tsv`) in the first split of data ("TAC_1"). Now write code to verify that only the following three types of labels are present in all the five splits of data: "B", "I", and "O".

### TASK 5.2 : (Pre-processing the Data)

This most challenging part of utilizing a platform like Hugging Face is to pre-process the data in the right format and data structure to ensure the hugging face models can access, read and fine-tune your data perfectly and give the best results.

This hugging face tutorial page gives some direction as to how to prepare your custom dataset to be utilized in a huggin face model: https://huggingface.co/transformers/v3.2.0/custom_datasets.html. Please go through this to get some direction on how to perform this. But remember we are preparing data for a specific purpose, i.e. **Token Classification**. So your data needs to be prepared in a specific way that might be different from data preparation for other NLP tasks like Translation, Summerization, etc.

Although this task can be performed in a variety of ways, the best **suggested path** would be to closely understand the internal hugging face `ncbi dataset` used in the above example (or any internal hugging face dataset you like), and generate an exactly similar data structure that would contain data from TAC2017: `MasterThesis_DS/biobert_ner/TAC2017_BioBERT_examples/TAC_1`

This would involve the following steps:

1. Read data from train and test tsv files in: `MasterThesis_DS/biobert_ner/TAC2017_BioBERT_examples/TAC_1`
2. Split train into train and validate. Use any percent split you think best.
3. Generate a data structure same as the internal hugging face `ncbi dataset` data structure containing all the TAC_1 data just extracted above.

### TASK 5.3: Train and Evaluate

Fine-tune the bioGPT model you saved in your hugging face git using this custom ADR dataset you just created. Also show its evaluation results on the ADR test set using `seqeval` as before.  

![image](https://github.com/travislatchman/Name-Entity-Recognition-using-BioGPT/assets/32372013/621510d0-e45a-49a1-889d-598b3e482e49)  

Precision: 0.312210200927357  
Recall: 0.10401647785787847  
F1-score: 0.1560448049439938  
Accuracy: 0.9313810222499146  


## Use BioGPT as feature extractor and construct your own BioGPT for token classification, and fine-tune it for NER

Generative model BioGPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence.When it comes to the token classification task, however, like we have mentioned, we do not have a BioGPTForTokenClassification class prepared for token classification in HuggingFace, and you may have encountered warnings when loading BioGPT weights to GPT2ForTokenClassification.

Luckily, as in https://huggingface.co/microsoft/biogpt, we could use BioGPT as a text feature extractor. In the tutorial, two models, BioGptForCausalLM and BioGptModel are provided.

We can't use BioGptForCausalLM to get the features as it returns logits for next token prediction.  But in the last_hidden_state attributes the BioGptModel provides, we can get a 1024*1 feature for each token.

Taking advantage of BioGptModel(**input).last_hidden_state, construct a new model for a 3-class classification, and load the pretrained BioGPT weights. Basically, we will add a set of fc layers to decrease the feature dimensionality from 1024 to 3.

The new model needs to inheriate all the attributes from BioGptModel(). To show this, name your model as My_BioGPT_Token_Classification, and run the blocks provided.

implement a custom training loop to fine-tune the model. modify the training loop, loss function, and evaluation to work with custom model.
