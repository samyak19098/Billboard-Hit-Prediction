Project made as a part of CSE343: Machine Learning course
# Billboard-Hit-Prediction
[Hit Song Science](https://en.wikipedia.org/wiki/Hit_Song_Science) concerns with the possibility of predicting whether a song will be a hit, before its distribution using automated means such as machine learning software. This motivated us to dig deeper to unravel how different audio features would help in predicting if a song would feature in the Billboard Top 100 Chart and build a two way usability model - both to the musicians composing the music and the labels broadcasting it.The project also aligns with our team’s vision of exploring real world applications of machine learning techniques and making them useful in common domains. We explore prediction models on data from [MSD](http://millionsongdataset.com/), Billboard and Spotify using ML techniques.
# Introduction
With the forever-expanding music industry and the number of people keen on listening to popular music, it becomes very important to come up with a classifier that can predict whether a song is ‘hit’ or ‘non-hit’ to help the musicians and the music labels. Using the data collected from Billboard, Spotify and Million Song Dataset, our project takes into account several features for a song like audio features and related artist data and based on that uses Machine Learning based classification algorithms to develop models that can help us achieve the desired classification. Our goal is to make accurate as well precise predictions. Our models will indicate what choices of a particular feature make a positive impact so that the musicians and music engineers can plan accordingly to give their songs the best chance of being classified as ‘hit’. We include both low-level and high-level analysis. Low-level analysis is done using the audio data and extracting raw audio features like spectrograms to train models. High-level analysis includes using high-level human understandable features like danceability, loudness, and acousticness.

# Data
Link to dataset : https://drive.google.com/drive/folders/1W7jRJYta_x7VoLn1MjbPKUqhpEmA_U_a?usp=sharing
</br>
Download the dataset for High-Level-Classification and Low-Level-Classification into the 'Data' folder in both the folders before running the models.

# Details
## Final Report for the project : [Final Report](https://drive.google.com/file/d/1lfrveMOsT5MYvtjfqLwC6ay6oVqJKHwL/view?usp=sharing)
### 1. High-Level-Classification
Complete code (with Preprocessing, EDA and models) is contained in 'main_code.ipynb'. Plots for EDA is in the 'EDA' folder. All the pickled models with their plots are saved in their respective folder in the 'models' folder.<br/>
To load a pickled model :
```python
file = open(PATH, "rb")
model = pickle.load(f)
file.close()
```
### 2.Low-Level-Classification
Model training and testing codes are in 'training.py' and 'testing.py' respectively. Pickled models are saved in the 'plots' folder. Loss plot and confusion matrix plots are in the 'plots' folder.<br/>
To load model :
```python
model = AudioClassifier()
model.load_state_dict(torch.load(PATH))
```
