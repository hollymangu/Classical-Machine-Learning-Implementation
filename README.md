# Machine Learning Assignment 1: Classical Machine Learning 🧮
Implementing Classical Machine Learning solutions in Python using the Scikit-Learn library and other libraries , specifically classification methods applied to the Digits Dataset.

# Libraries/  Languagaes
- Python
- Visualisation Packs: Pandas, Numpy, Matplotlib, Seaborn
- Classification Methods: Logistic Regression, Gradient Boosting, Decision Tree, SVM, Naive Bayes, LDA, Random Forest, KNN

# Findings
Model Performance Comparison

<img width="664" height="175" alt="image" src="https://github.com/user-attachments/assets/1dcae38c-47d2-406e-8ee9-bc3b1618b2a7" />

# The Process
*Loading and EDA*

This project  begins with loading the scikit-learn digits dataset, containing 1,797 grayscale images of handwritten digits (0-9). Each image is 8×8 pixels, flattened into a 64-dimensional feature vector where each value represents pixel intensity (0-16). The dataset is  balanced across the 10 classes, making it suitable for standard accuracy metrics without immediate need for class rebalancing techniques.

*Data Prep and Splitting*

The data is then split into training (80%) and testing (20%) sets using train_test_split with a fixed random seed (42) for reproducibility. Producing training samples and 360 test samples. No additional preprocessing(e.g normalization) is applied since pixel values are already in a consistent range (0-16).

*Algorithm Selection and Implementation*

Eight distinct classification algorithms are implemented, representing diverse machine learning paradigms. 

**Linear Models**: Logistic Regression and LDA capture linear decision boundaries.
**Tree-Based Models**: Decision Trees, Random Forests, and Gradient Boosting handle non-linear relationships through hierarchical splitting and ensemble methods.
**Distance-Based**: k-NN classifies based on similarity in feature space.
**Probabilistic**: Naive Bayes applies Bayesian probability with feature independence assumption.
**Maximum Margin**: SVM finds optimal separating hyperplanes.

Each algorithm is first implemented in its default configuration to establish baseline performance before optimization.

*Dimensionality Reduction Exploration*

Dimensionality Reduction involves cutting down the number of features (variables/dimensions) in a dataset while keeping essential information, transforming high-dimensional data into a simpler, lower-dimensional space to prevent overfitting, speed up training, and make data easier to visualize and interpret.

PCA (Principal Component Analysis) is  a core technique in dimensionality reduction, it identifies components that best capture data spread, allowing for data compression, noise reduction, visualization of patterns, and improved model efficiency by removing redundant features.  

In this project, we experimentally applied to Logistic Regression, Random Forest, and k-NN to assess whether reducing 64 dimensions improves performance or efficiency. Results show PCA with 30 components slightly reduces Logistic Regression accuracy (97.22% vs 97.50%), while aggressive reduction to 2 components severely harms Random Forest and k-NN (dropping to ~60% accuracy). This finding suggests the original 64 features contain meaningful variance for classification that shouldn't be overly reduced.#

*Hyperparameter Tuning*

Each classifier underwent systematic optimization using GridSearchCV with 5-fold cross-validation to find the best configuration. This included tuning Logistic Regression's regularization strength, Gradient Boosting's tree parameters, SVM's kernel settings, and all other algorithms' specific hyperparameters. The process identified SVM with C=0.98 and gamma=0.00095 as the optimal configuration, demonstrating how methodical tuning improves model performance.

*Multi-Metric Performance Evaluation*

After optimization, models were evaluated using five complementary metrics: accuracy, balanced accuracy, confusion matrices, precision-recall curves, and ROC analysis. This revealed performance tiers with SVM leading (98.89% accuracy, 0.9999 AUC), followed by k-NN and ensemble methods, while Decision Trees and Naive Bayes performed less effectively. The comprehensive evaluation provided actionable insights, recommending SVM for deployment while noting k-NN and Gradient Boosting as viable alternatives.

# WWW

-Model selection and diversity: As a team we successfully implimented a range of classification algorithms to a singular data set, and created a  comparative analysis whilst building and understanding of different machine learning paradigms.

-Choice of hyperpaprementer tuning: Implementing GridSearchCV allowed for analysis beyond default parameters, leading to tangible performance gains (e.g., fine-tuning SVM to 98.89% accuracy) and ensured models were performing near their optimal potential.

-Comprehensive Analysis: Balanced Accuracy, Confusion Matrices, ROC-AUC, Precision-Recall curves). This provided a holistic view of model performance, highlighting strengths in class discrimination (excellent AUC scores) and identifying specific failure modes between similar digits.

-Clear Performance Hierarchy and Insightful Interpretation: The results cleanly categorized models into performance tiers. As well as this, it correctly interpreted the 'why' — noting SVM's success to margin separation, k-NN's to local similarity, and Naive Bayes' limitations to its violated independence assumption in image data.

# EBI

- Deeper Dive in Dimensionality ReductionL: Utalising PCA against other techniques like t-SNE or LDA for visualization to better understand the feature space
  
- Addressing Computational Efficiency: If analysis also considered model trade-offs, such as comparing the training/prediction speed and model size of SVM vs. k-NN vs. Random Forest. It would allow factoring for real-world deployment scenarios.
  
- Error Analysis and Misclassification Focus: Including an analysis visualizing the actual digit images that models consistently misclassified could reveal if errors are due to ambiguous handwriting, and if they are consistent across all top models.

- Validation Stratergy Improvement: As Hyperparameter tuning was done on the same training set, having a dedicated validation set or nested cross-validation could ensure the tuned hyperparameters generalize perfectly to the held-out test set, providing a more rigorous estimate of final performance.
