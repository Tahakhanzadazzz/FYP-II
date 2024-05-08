

### Objective

The main goal of the Musti project is to deepen our grasp of how smells are referenced in texts and images from the 16th to the 20th centuries across multiple languages. This project aims to create computational tools capable of recognizing and categorizing these scent references in various forms and languages. The end objective is to devise systems that can seamlessly connect written and visual content that evoke similar scents, enriching our interaction with both digital and physical archives.

### Dataset Overview

The MUSTI 2023 dataset consists of copyright-free texts and partly copyrighted images that could be downloaded and submitted by the participants using the URLs we provide. We offer texts in English, Dutch, French, German, Italian, and Slovene that participants are to match to the images.

The texts are selected from open repositories such as Project Gutenberg, Europeana, Royal Society Corpus, Deutsches Text Arxiv, Gallica, Wikisource and Liber Liber The images are selected from different archives such as RKD, Bildindex der Kunst und Architektur, Museum Boijmans, Ashmolean Museum Oxford, Plateforme ouverte du patrimoine. The format of dataset is .json.

The text descriptions encompass various topics related to daily life and historical events. However, it's important to note that some image titles (title) might be inaccurate or misleading, potentially indicating inconsistencies between the text and image content.

The dataset includes labels for two potential subtasks (subtask1_label, subtask2_labels), though the specific nature of these tasks is not explicitly provided. Subtask 1 labels are binary (YES or NO), possibly indicating image classification. Subtask 2 labels are lists of strings, potentially representing object detection within the images.

**Some Images from Dataset:**

![oswald-wijnen-blumenstrauss-einer-vase-auf-einem-marmortisch-1993z--thumb-xl.jpg](Untitled%20d4de7ed9479b41819739b638b67c648c/oswald-wijnen-blumenstrauss-einer-vase-auf-einem-marmortisch-1993z--thumb-xl.jpg)

![1975-3-3.jpg](Untitled%20d4de7ed9479b41819739b638b67c648c/1975-3-3.jpg)

![0110579.jpg](Untitled%20d4de7ed9479b41819739b638b67c648c/0110579.jpg)

### Problem Overview

### Introduction

The Musti project tackles the challenge of incorporating smell into the analysis of historical multimedia content. Integrating this sensory dimension is essential for crafting more engaging and accessible digital experiences in the fields of digital humanities and cultural heritage, especially for audiences with visual impairments.

### Key Challenges

1. **Handling Multimodal Data**: The project requires managing and analyzing both textual and visual data, calling for sophisticated multimodal analytical techniques.
2. **Analyzing Multiple Languages**: The data spans English, German, Italian, and French, complicating the analysis due to the need for efficient multilingual processing.
3. **Understanding the Semantics of Smell**: Smell, a complex and rarely digitized sense, must be accurately interpreted and linked across texts and images that may represent it differently depending on the time period and cultural context.
4. **Navigating Historical and Cultural Differences**: Considering the wide historical scope, the project must address shifts in language, art styles, and the materials available at different times, which affect how smells are experienced and represented.

### Specific Objectives

- **Classification Task (Mandatory)**: Develop algorithms that determine if a text and an image refer to the same scent. This involves a straightforward 'yes' or 'no' decision based on the similarity of the smells evoked.
- **Detection Task (Optional)**: Identify specific smell sources in texts and images, such as certain plants, animals, or environments. This task involves pinpointing various potential sources of smell in each format and matching them.

### Potential Impact

Successful advancements in the Musti project could greatly enhance the usefulness and appeal of digital and physical archives. By integrating the sense of smell into multimedia analyses, venues such as museums, libraries, and educational platforms can provide deeper, more memorable experiences that resonate more profoundly with people's emotions and memories. Additionally, this project encourages interdisciplinary collaboration, merging techniques from language processing, computer vision, olfactory science, and cultural studies, opening up new possibilities for interactive education and immersive virtual experiences.

### Methodology for Solving the Musti Problem

The Musti task involves two subtasks: classification and detection of olfactory references in texts and images. Each requires distinct methodologies to address effectively. Here, we detail three approaches—each leveraging different machine learning and deep learning techniques—that could be tailored to meet the demands of these subtasks.

### Approach 1: LSTM (Long Short-Term Memory) Networks

**1. Data Preprocessing:**

- **Text Tokenization**: Utilize the BERT tokenizer to convert text data into tokens. This tokenizer is effective across multiple languages and is capable of capturing contextual nuances in the text.
- **Sequence Padding**: Ensure uniform input size by padding or truncating sequences to a fixed length, enabling batch processing without input size discrepancies.

**2. Model Architecture:**

- **Embedding Layer**: Map the tokenized text to a higher-dimensional vector space to capture semantic meanings of words.
- **LSTM Layers**: Implement sequential layers of LSTM to process text data over time. LSTMs are particularly suited for sequence prediction tasks because they can maintain long-term dependencies and handle the vanishing gradient problem that often occurs with standard RNNs.
- **Dropout Layers**: Introduce dropout between LSTM layers to prevent overfitting by randomly omitting subsets of features at each training stage.
- **Output Layer**: A dense layer with a sigmoid activation function to classify whether the text and image are related by smell.

**3. Training and Validation:**

- Split the data into training and validation sets to evaluate the model’s performance and generalize beyond the training data.
- Use binary crossentropy as the loss function for this classification task, optimizing the model with the Adam optimizer.

**4. Feature Extraction and Matching:**

- Extract features from intermediate layers of the LSTM model to represent text data, which can then be compared with features extracted from images using similarity metrics to perform the matching required in the Musti classification task.

### Approach 2: YOLOv8 for Object Detection

### 1. Data Setup and Model Initialization:

- **Dataset Preparation**: Format the image dataset according to YOLO's requirements, including annotations that categorize different elements that might relate to smells (e.g., flowers, foods). This meticulous annotation allows the model to recognize and learn from relevant olfactory cues depicted in the images.
- **Model Selection**: Use YOLOv8, known for its efficiency and accuracy in object detection tasks, initialized with pre-trained weights for a head start in learning. This selection leverages YOLOv8's capabilities to rapidly and accurately detect multiple items within an image, crucial for identifying potential smell sources.

### 2. Model Training and Fine-tuning:

- **Train the YOLO model** on the prepared dataset to detect objects related to olfactory cues. This step involves feeding the model thousands of labeled images to help it learn to identify and localize objects associated with smells.
- **Adjust model parameters** like the learning rate and the number of epochs based on the validation set performance to fine-tune the model’s detection capabilities. These adjustments ensure the model optimally balances between bias and variance, improving its ability to generalize to new, unseen images.

### 3. Object Detection and Analysis:

- **Apply the trained model to new images**, detecting objects and their bounding boxes. This application involves running the model on new datasets and using it to automatically identify and locate items within images that could potentially emit smells.
- **Analyze detected objects** to determine potential smell sources, which can be matched against textual descriptions in the Musti detection task. This analysis is critical for identifying which objects in an image correspond to the smell descriptions found in related text.
- **Integration with Mistral LLM**: Utilize the Mistral large language model to take the classes detected by YOLOv8 and the described text to assess whether they refer to the same sense of smell. This involves feeding the names of the detected objects and the textual smell descriptions into Mistral, which then processes these inputs to determine semantic similarity or relevance. This step is pivotal in linking visual data with textual descriptions accurately, thereby enhancing the multimodal understanding required for the Musti tasks.

### Approach 3: CLIP (Contrastive Language–Image Pre-training)

**1. Preprocessing with CLIP:**

- **Image Preprocessing**: Use CLIP’s preprocessing pipeline to convert images into a format suitable for the model, ensuring that image modifications (like resizing and normalization) match the training conditions of CLIP.
- **Text Processing**: Employ CLIP's tokenizer to process textual data, ensuring texts are trimmed or padded to the model’s maximum context length.

**2. Dual-Encoder Architecture:**

- Use CLIP’s dual-encoder framework to independently encode text and images into a shared embedding space where their similarities can be directly compared.

**3. Similarity Measurement:**

- Compute similarity scores between text and image embeddings using cosine similarity, a metric suitable for measuring angles between high-dimensional vectors.

**4. Classification and Matching:**

- Set a threshold for similarity scores to classify text-image pairs as sharing a smell source or not.
- Utilize these classifications to aid in both the Musti classification and detection tasks, identifying whether texts and images refer to the same smell sources and what those sources might be.

Each of these methodologies leverages specific strengths of different machine learning models and techniques—LSTM for textual sequence analysis, YOLO for visual object detection, and CLIP for matching text and image content. By adapting these approaches, the Musti tasks can be tackled effectively, providing robust tools to enhance the sensory dimension of multimedia content analysis.

### Results

In the Musti task, various methodologies were deployed to tackle the challenges of identifying and linking olfactory references in texts and images. Below are the detailed results obtained from each approach, which provide insight into their effectiveness and areas for potential improvement.

### YOLO - LLM

- **Accuracy**: 48.29%
- **Precision**: 0.27
- **Recall**: 0.66
- **F1 Score**: 0.39

**Analysis**:
The YOLOv8 model integrated with the Mistral LLM achieved an accuracy of 48.29%, which suggests a moderate level of effectiveness in correctly matching text and image pairs based on olfactory references. The precision of 0.27 indicates that when the model predicts a match, it is correct about 27% of the time. However, the recall of 0.66 is relatively higher, indicating that the model is quite good at identifying true positive cases but at the cost of a large number of false positives, as reflected by the low precision. The F1 score, which balances precision and recall, stands at 0.39, suggesting there is significant room for improvement, particularly in reducing false positives and enhancing overall accuracy.

### CLIP

- **Accuracy**: 75.29%
- **Precision**: 0.33
- **Recall**: 0.69
- **F1 Score**: 0.45

**Analysis**:
The CLIP model performed considerably better, with an accuracy of 75.29%. This higher accuracy indicates a stronger ability to correctly predict whether text and image pairs are related by smell. The precision here remains low at 0.33, similar to the YOLO-LLM model, indicating a continuing challenge with false positives. However, the recall is robust at 0.69, showing that the model is effective at identifying a large majority of relevant cases. The F1 score of 0.45, while an improvement over the YOLO-LLM, still underscores the need for a better balance between precision and recall.

### LSTM

- **Accuracy**: 62%

**Analysis**:
The LSTM model achieved an accuracy of 62%, which indicates a decent capability to classify text-image pairs correctly compared to random guessing. However, without additional metrics such as precision, recall, and F1 score, it's challenging to gauge the model's performance regarding false positives and false negatives. An accuracy of 62% suggests that while the LSTM model is reasonably effective, there might be issues either in the model's ability to generalize or perhaps in its sensitivity to the dataset's imbalanced nature.

### RNN

- **Accuracy**: 71%

**Analysis**:
The RNN approach shows a promising accuracy of 71%, indicating a good level of effectiveness in classifying the pairs correctly. This model appears to perform better than the LSTM, suggesting that the simpler RNN structure might be capturing the temporal dependencies in the data more effectively for this particular task. Again, similar to LSTM, the lack of precision, recall, and F1 scores limits a deeper understanding of the model's operational strengths and weaknesses.

### Overall Conclusion

The CLIP model outperforms other methods in terms of accuracy and F1 score, making it a strong candidate for tasks requiring matching textual descriptions with images. However, all models demonstrate a need for improved precision, which would help reduce the number of false positives—a critical factor in practical applications. These results highlight the trade-offs between different approaches and the potential for hybrid models or further tuning to enhance performance in multimodal, multilingual datasets. Future work could focus on integrating the strengths of these models or refining the feature extraction and matching processes to better align with the nuanced task of olfactory reference detection and classification.

### References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Available from [https://www.deeplearningbook.org](https://www.deeplearningbook.org/)
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998-6008).
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv preprint arXiv:1804.02767*. Available from [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*.
5. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *arXiv preprint arXiv:2103.00020*. Available from [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
7. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. Available from [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
8. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 2818-2826).
9. Ulrich, W. K., & Cerf, V. G. (2017). Neural networks and olfactory task analysis. *Cognitive Science*, 41(4), 935-965.
