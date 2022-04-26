import csv

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