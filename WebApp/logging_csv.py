import csv
import pandas as pd

def logging_csv(number, mode, landmark_list):
    #print(mode)
    #print(number)
    if mode == 1:
       # print("here")
        csv_path = 'model/keypoint_classifier/keypointTempApp.csv'
        with open(csv_path, 'a', newline="") as f:

            #print(landmark_list)
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def getLastRowValue():
    
    
    df = pd.read_csv('model/keypoint_classifier/keypoint.csv')
    last_number=df.iloc[-1,0]
    print(last_number)
    return last_number