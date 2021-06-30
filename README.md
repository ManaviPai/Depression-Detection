# Depression-Detection Using a Chatbot

### This project was built during the Covid-19 times to help people who suffer from depression to detect this mental illness with the help of a chatbot. The emotion detection model was built using CNN to help in the detection of emotions using the ISEAR dataset. The chatbot however used Keras's Sequential Model with an SGD optimizer to help predict an appropriate response for a User Text.

## Requirements:
1. Python 3.7
2. Keras 2.2.4
3. Tensorflow 2.1.0
4. NLTK 3.6.2
5. numpy 1.19

## Files:
1. training.py : Training chatbot to predict appropriate responses
2. Model_CNNfin: The emotion detection model using CNN
3. app.py : The Flask Application that runs the chatbot while detecting the emotion of the user's response for each text.
4. model.h5 : The saved Sequential for the Chatbot's responses
5. cnn_model.h5 : The saved emotion detection model used to detect user's emotion.

## How to run project:
1. Clone the Github Repo
2. Create a virtual environment
3. Install the required libraries
4. Run app.py file 
