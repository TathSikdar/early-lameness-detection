import cv2
import numpy as np
import datetime as dt

class Cow:
    def __init__(self, session):
        #Key Cow information
        self.c_session = dt.datetime.now()
        self.c_id = 0                               #Cow identification for session
        self.c_timestamp =  dt.datetime                 #Timestamp of cow idenfication
        self.c_cow_id = "null"                      #Ear Tag identification
        self.c_ear_tag_confi = 0.0                  #Confidence in Ear Tag Number
        self.c_prediction = 0                       #Lameness Prediction
        self.c_confidence = 0.0                     #Confidence in Lameness Prediction
        self.c_needs_review_flg = False             #Flag for human verification
        self.c_review_prio = "null"                 #Priority for human verification
        self.c_corrected_flg = False                #Flag if in queue to be corrected
        self.c_corrected_value = 0                  #Human corrected Lameness value
        self.c_correction_timestamp = dt.datetime       #Human correction timestamp
        self.c_model_version = "V0.0"               #Lameness Model number
        
    def update_session(self, session: dt.datetime):
        self.c_session = session
        
    def update_id(self, c_id: int):
        self.c_id = c_id
        
    def update_cow_id_and_confi(self, ear_tag: str, confidence: float):
        self.c_cow_id = f"cow_{ear_tag}"
        self.c_ear_tag_confi = confidence
        
    def update_prediction_and_confi(self, prediction: int, confidence: float):
        self.c_prediction = prediction
        self.c_confidence = confidence
        
        #Calculate the new review flag and priority
        self.calculate_needs_review()
        self.calculate_review_prio()
    
    def calculate_needs_review(self):
        #Add logic later
        pass   
    def calculate_review_prio(self):
        #Add logic later
        pass
        
    def update_correction(self, new_score: int, time: dt.datetime):
        self.c_corrected_value = new_score
        self.c_corrected_flg = True
        self.c_correction_timestamp = time
        
    def update_model(self, version):
        self.c_model_version = version
        
print("-----------------------------")
print(f"dt.date type:           {dt.date}")
print(f"dt.datetime type:       {dt.datetime}")
print(f"dt.datetime.now() type: {type(dt.datetime.now())}")
print(f"dt.time type:           {dt.time}")
print("-----------------------------")


print(f"dt.time.max(): {dt.time}")


print("-----------------------------")
print(type(10))