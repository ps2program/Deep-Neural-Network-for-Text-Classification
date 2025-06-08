
# Running in Google Colab

When running this project in Google Colab:

1. **Data Storage**:
   - The Yelp dataset will be downloaded to `/content/data/yelp_review_polarity_csv/`
   - This is in Colab's temporary runtime storage
   - Data will be lost when the runtime is disconnected

2. **Model Storage**:
   - Trained models will be saved to `/content/models/`
   - This is also in Colab's temporary runtime storage
   - Models will be lost when the runtime is disconnected

3. **To Persist Data and Models**:
   - Use Google Drive to save important files:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Save model to Google Drive
   model.save('/content/drive/MyDrive/your_model.keras')
   
   # Load model from Google Drive
   model = tf.keras.models.load_model('/content/drive/MyDrive/your_model.keras')
   ```

4. **Downloading from Colab to Local Machine**:
   ```python
   # Download a single file
   from google.colab import files
   files.download('/content/models/best_model.keras')  # Downloads model file
   files.download('/content/data/yelp_review_polarity_csv/train.csv')  # Downloads data file
   
   # Download entire directories (zip them first)
   !zip -r models.zip /content/models/
   !zip -r data.zip /content/data/
   files.download('models.zip')
   files.download('data.zip')
   ```

5. **Setup in Colab**:
   ```python
   # Clone the repository
   !git clone https://github.com/ps2program/Deep-Neural-Neural-Network-for-Text-Classification.git
   %cd Deep-Neural-Network-for-Text-Classification
   
   # Install dependencies
   !pip install -r requirements.txt
   ```
