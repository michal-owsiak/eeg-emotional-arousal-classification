## **EEG-based Binary Classification of Emotional Arousal: Exploratory Analysis and Machine Learning on DEAP Dataset**

This repository contains a [Jupyter Notebook](https://github.com/michal-owsiak/eeg-emotional-arousal-classification/blob/main/index.ipynb) that presents a comparative study of neural network and classical machine learning algorithms applied to the binary classification of EEG data. The aim was to assess how well various models can distinguish between two mental states based solely on bandpower features extracted from EEG signals.

The notebook demonstrates a complete end-to-end workflow, including preprocessing, feature extraction, model training, evaluation, and visualization.

---

## **Abstract**

**Background**:  
Electroencephalography (EEG) provides a non-invasive window into brain activity and is increasingly explored in affective computing. Emotional arousal — a key dimension of emotional experience — can potentially be inferred from EEG features, though its neural signatures remain subtle and difficult to extract.

**Aims**:  
This study evaluates the feasibility of classifying binary emotional arousal (high vs. low) based on EEG recordings from the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html). It was aimed to assess how well bandpower features from multiple frequency bands and scalp locations can distinguish arousal levels using both neural and classical machine learning classifiers.

**Methods:**  
EEG signals from 32 participants, each exposed to 40 music video trials (63 seconds each), were segmented into overlapping windows of 5 seconds with a 2.5-second step (50% overlap). For each window, absolute and relative bandpower was computed across four frequency bands (theta, alpha, beta, gamma) from 32 EEG channels, yielding 256 features per window. The resulting dataset contained 30,720 labeled windows (training + test). Feature selection was performed using random forest importance scores, and the top 50 features were used to train a neural network, along with logistic regression, random forest, SVM, and XGBoost classifiers. Performance was evaluated using accuracy and ROC AUC metrics.

**Results:**  
All models achieved modest predictive performance, with test accuracy ranging from 60% to 63% and ROC AUC ranging from 0.62 to 0.67. The SVM classifier achieved both the highest accuracy (63%) and the highest ROC AUC (0.67). Trial-level permutation tests were subsequently performed to assess whether these scores exceeded chance-level performance. Both accuracy and ROC AUC were significantly above chance for all models after Holm correction for multiple comparisons (adjusted *p* < 0.05).

**Conclusions:**  
The results suggest that EEG bandpower features carry only limited and hard-to-extract information about arousal, leading to modest model performance across classifiers. These findings highlight both the potential and limitations of EEG-based emotion recognition and suggest the need for more sophisticated feature engineering or multimodal data to improve performance in affective computing tasks.

---

## **Tools used**

**Python libraries and packages:**
```
> TensorFlow / Keras
> scikit-learn
> XGBoost
> SciPy
> NumPy
> pandas
> matplotlib
> seaborn
> gdown
> os
> IPython
```
## **Data preparation**

The EEG data were preprocessed into numerical features as follows:

- **Bandpower computation:** Absolute and relative power calculated for θ, α, β, and γ bands for each of 32 EEG channels.
- **Data splitting:** Train/test split with stratified sampling.
- **Scaling:** Standardization applied to features prior to model training.

## **Modeling and evaluation**

### **Neural Network (MLP)**  
A shallow dense network (multi-layer perceptron) with regularization and droput was trained using early stopping to avoid overfitting.  


### **Classical Machine Learning Models**  
Four traditional models were tested for comparison:  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  

Each model was evaluated using accuracy, ROC AUC, precision, recall, and F1-score. Above-chance performance (for accuracy and ROC AUC) was assessed using one-sided, within-participant trial-level permutation tests with Holm correction for multiple comparisons.

**Visualizations include:**
- Raw sample EEG signal along with spectrogram
- Histogram depicting distribution of the arousal self-assesments
- EEG bandpower and feature importance heatmaps
- Accuracy and loss curves (for MLP) 

---

## **Results and conclusions**

| Model                | Accuracy | ROC AUC | Accuracy *p* (Holm) | ROC AUC *p* (Holm) |
|----------------------|----------|---------|---------------------|--------------------|
| Neural network (MLP) | 0.61     | 0.65    | 0.0144              | 0.0152             |
| Logistic regression  | 0.60     | 0.62    | 0.0032              | 0.0220             |
| Random forest        | 0.61     | 0.65    | 0.0024              | 0.0020             |
| SVM                  | 0.63     | 0.67    | 0.0020              | 0.0020             |
| XGBoost              | 0.60     | 0.64    | 0.0030              | 0.0020             |

*p*-values compare each metric against chance using one-sided, within-participant trial-level permutation tests, with Holm correction across all 10 model/metric tests.

---

### **Key observations:**

- All models achieve **statistically significant above-chance performance** in both accuracy and ROC AUC, as confirmed by trial-level permutation tests with Holm correction (adjusted *p* < 0.05), although their overall predictive performance remains modest.
- **SVM** achieves both the highest **accuracy (0.63)** and the highest **ROC AUC (0.67)**.
- The **neural network (MLP)** and **random forest** achieve comparable point estimates: accuracy of 0.61 and ROC AUC of 0.65 when rounded to two decimal places.
- Accuracy ranges from 0.60 to 0.63 and ROC AUC from 0.62 to 0.67, indicating modest predictive performance across models.
- The permutation tests compare each model against chance, not against other models; they do not establish whether the differences between algorithms are statistically significant.

---

Overall, the results suggest that more advanced feature engineering, temporal modeling, or deep architectures (e.g., CNNs, RNNs) may be necessary to meaningfully improve classification performance in this domain.

---

**Author of the notebook**  

Michał Owsiak
