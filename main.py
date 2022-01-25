from tkinter.dialog import DIALOG_ICON
from typing import Dict
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP

face_classifier = cv2.CascadeClassifier(r'D:\Emotion_Detection_CNN-main-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\Emotion_Detection_CNN-main-main\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# cap = cv2.VideoCapture(0)



# Hello Shashikant



Dict = defaultdict(lambda:0)

def solve():
    cap = cv2.VideoCapture(0)
    while True:
        # Dict = {}
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                # Storing the frequency of the emotion detected.
                Dict[label]+=1
                # print(label)
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Facqes',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        # print(labels)
        # Printing all the emotions detected and their frequency.
        # print(Dict)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cap.release()
    # cv2.destroyAllWindows()

# print(Dict)

# Main Function for scraping
def main(emotion):
  
    # IMDb Url for Drama genre of
    # movie against emotion Sad
    if(emotion == "Sad"):
        urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Musical genre of
    # movie against emotion Disgust
    elif(emotion == "Disgust"):
        urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Family genre of
    # movie against emotion Anger
    elif(emotion == "Anger"):
        urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Thriller genre of
    # movie against emotion Anticipation
    elif(emotion == "Anticipation"):
        urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Sport genre of
    # movie against emotion Fear
    elif(emotion == "Fear"):
        urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Thriller genre of
    # movie against emotion Enjoyment
    elif(emotion == "Enjoyment"):
        urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Western genre of
    # movie against emotion Trust
    elif(emotion == "Trust"):
        urlhere = 'http://www.imdb.com/search/title?genres=western&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Film_noir genre of
    # movie against emotion Surprise
    elif(emotion == "Surprise"):
        urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'

    else:
        # https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc
        urlhere = 'https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc, asc'
    # HTTP request to get the data of
    # the whole page
    response = HTTP.get(urlhere)
    data = response.text
  
    # Parsing the data using
    # BeautifulSoup
    soup = SOUP(data, "lxml")
  
    # Extract movie titles from the
    # data using regex
    title = soup.find_all("a", attrs = {"href" : re.compile(r'\/title\/tt+\d*\/')})
    return title
  





if __name__ == "__main__":
    print ("Executed when invoked directly")
    solve()

    emotion_name = ""
    maxi = -1000000
    for i in Dict:
        if Dict[i] > maxi and i != "Neutral":
            emotion_name = i;
            maxi = Dict[i]
    print("Emotion Name : ",emotion_name) 


    a = main(emotion_name)
    count = 0
  
    if(emotion_name == "Disgust" or emotion_name == "Anger" or emotion_name =="Surprise"):
        for i in a:
  
            # Splitting each line of the
            # IMDb data to scrape movies
            tmp = str(i).split('>;')
  
            if(len(tmp) == 3):
                print(tmp[1][:-3])
  
            if(count > 13):
                break
            count += 1
    else:
        for i in a:
            tmp = str(i).split('>')
  
            if(len(tmp) == 3):
                print(tmp[1][:-3])
  
            if(count > 11):
                break
            count+=1

# solve()


# cap.release()
# cv2.destroyAllWindows()
# cap.release()
# print(Dict)
# cap.release()
# cv2.destroyAllWindows()
# print(Dict)

cap.release()
cv2.destroyAllWindows()
