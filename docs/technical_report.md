# Audio Emotion Recognition: Technical Report

## 1. Introduction & Problem Definition

Emotion recognition from speech is a crucial component in human-computer interaction systems. The ability to accurately detect emotional states from voice enables more natural and empathetic interactions in applications ranging from virtual assistants to mental health monitoring. This project implements and evaluates an audio-based emotion recognition system using classical machine learning approaches.

**Research Objectives:**
1. Develop a robust feature extraction pipeline for capturing emotion-relevant acoustic characteristics
2. Evaluate and compare traditional machine learning algorithms for emotion classification
3. Identify the most informative acoustic features for emotion recognition
4. Analyze system limitations and propose improvements

## 2. Data Acquisition & Preprocessing

### Dataset

The system was developed and evaluated using the CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset), which contains:
- 7,442 original audio files of emotional speech
- 91 actors (48 male, 43 female) with diverse ethnic backgrounds
- 6 emotion categories: anger, disgust, fear, happy, neutral, and sad
- Sentences spoken with different emotional intensities

### Preprocessing Pipeline

Our audio preprocessing pipeline consists of:
1. **Noise Reduction**: Using spectral gating to remove background noise
2. **Amplitude Normalization**: Normalizing audio volume to ensure consistent feature extraction
3. **Silence Removal**: Trimming leading and trailing silence for more focused analysis
4. **Segmentation**: Dividing longer audio into segments to capture localized emotional cues

## 3. Feature Extraction & Selection

### Acoustic Feature Set

We extracted a comprehensive set of acoustic features known to correlate with emotional expression:
1. **Mel-Frequency Cepstral Coefficients (MFCCs)**: 13 coefficients plus their delta and delta-delta derivatives
2. **Spectral Features**: Spectral centroid, bandwidth, contrast, and rolloff
3. **Prosodic Features**: Zero-crossing rate, energy, and RMS energy
4. **Statistical Derivatives**: Mean, standard deviation, min, max, and range for each base feature

### Feature Selection

Feature selection was performed using ANOVA F-value to identify the most discriminative features. Analysis revealed:
1. MFCC-based features, particularly their statistical properties, dominated the top-ranked features
2. Spectral contrast features provided complementary information
3. Zero-crossing rate and spectral bandwidth offered additional discriminative power

## 4. Model Development & Evaluation

### Model Comparison

We implemented and evaluated two primary machine learning algorithms:
1. **Random Forest**: An ensemble of decision trees offering good interpretability
2. **XGBoost**: A gradient boosting framework known for high performance

Both algorithms were evaluated using 5-fold cross-validation to ensure robust performance estimation.

### Statistical Analysis

Statistical significance testing revealed:
- **XGBoost** significantly outperformed **Random Forest** (p = 0.012, paired t-test)
- XGBoost achieved ~56.5% accuracy (95% CI: 55.8-57.4%)
- Random Forest achieved ~54.1% accuracy (95% CI: 53.3-55.3%)

### Performance by Emotion

Confusion matrix analysis revealed:
- Strong performance in classifying **anger**, **disgust**, and **sadness**
- Poorer performance in distinguishing **happy** and **neutral** emotions
- Common misclassifications between **happy** and **disgust**, and **neutral** and **disgust**

## 5. Results & Discussion

### Key Findings

1. **Statistical Significance**: XGBoost consistently outperforms Random Forest across multiple metrics
2. **Feature Importance**: MFCC features dominate the feature importance rankings, particularly their statistical derivatives
3. **Emotion Differentiation**: Some emotion pairs (e.g., happy vs. neutral) remain challenging to distinguish
4. **Performance Context**: Our accuracy (~56.5%) is competitive with similar classical ML approaches on emotional speech datasets

### Limitations

1. **Feature Engineering Constraints**: Heavy reliance on MFCC features may limit the capture of certain emotional cues
2. **Temporal Modeling**: Limited incorporation of emotion dynamics over time
3. **Similar Emotion Confusion**: Difficulty distinguishing between acoustically similar emotions
4. **Dataset Bias**: Potential cultural and demographic biases in emotional expression

## 6. Future Directions

Based on our findings, we propose several directions for improvement:

1. **Advanced Modeling Approaches**:
   - Implement deep learning approaches (CNNs for spectrograms, RNNs for temporal dynamics)
   - Develop hierarchical classification for similar emotions

2. **Enhanced Feature Engineering**:
   - Develop specialized features for commonly confused emotions
   - Better incorporate temporal dynamics of emotional expression
   - Explore multimodal integration (audio + text) when available

3. **Data Improvements**:
   - Implement data augmentation techniques (pitch shifting, time stretching)
   - Cross-corpus validation with multiple emotional speech datasets
   - Explore synthetic data generation for underrepresented emotions

## 7. Conclusion

This project demonstrates a complete pipeline for audio-based emotion recognition using classical machine learning approaches. While achieving competitive performance, our analysis reveals both strengths and limitations in the current approach. The statistical significance of our model comparison provides strong evidence for preferring XGBoost in this application domain, while the identified limitations offer clear directions for future improvements.

The system's current performance makes it suitable for applications where approximate emotion recognition is sufficient, particularly for distinguishing between anger, disgust, and sadness. For applications requiring higher accuracy or better discrimination between happy and neutral states, the proposed improvements would need to be implemented. 