### Data Extraction
- Utilized PyMuPDF library for text extraction from PDFs.
- Considered OCR for potential improvement in text ordering (with coordinates info) by processing it as PDF images.(Not tried yet)

### Data Preprocessing
- Identifying and removing irrelevant elements, such as 'copyright' and numbers, by observing the parsed lines from pdfs in training data.
- Customized stop words based on training data for effective preprocessing.

### Rule-Based Model
- Used regex parsing to identify light-related synonyms and classifying initial positives.
- Incorporated additional rule-based checks for false positives: fixture words and look-aheads like 'with lights', 'for lights', etc.

### Logistic Regression Model
- light phrase from positively classified instances in the rule-based model used as a input feature
- Generated BERT embeddings for the light phrases to use as input features for logistic regression.
- Trained logistic regression model against a binary label for lighting (1) or non-lighting (0) objects.

### Model Performance
- Evaluated model performance on test data, achieving an accuracy of approximately 77.5%.

### Inference Pipeline Setup
- Deployed the model using Streamlit on Streamlit Cloud for user interaction.
- Used Google Cloud Storage for data storage and DVC for remote data tracking.
- Implemented a Streamlit web app that can process both PDF URLs and direct product descriptions - to infer luminosity.
- Display confidence scores and additional metadata for interpreting model prediction.

### Luminosity Classifier web app - 
https://luminosity-classifier.streamlit.app/
