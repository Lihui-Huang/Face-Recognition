#import OpenCV module
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import face_recognition

def logSurveillance(name):
    with open('Surveillance.csv','r+') as f:
        # Use f.readlines
        myDataList = f.readlines()
        nameList =[]
        # Use datetime.now
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S")
        f.writelines(f'\n{name},{dt_string}')

KNOWN_FACES_DIR = '/Users/can.b/Desktop/Face Recognition/Face_Recognition/Project4/Surveillance/known_faces'
UNKNOWN_FACES_DIR = '/Users/can.b/Desktop/Face Recognition/Face_Recognition/Project4/Surveillance/unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model


# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image - use face_recognition.load_image_file
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        print(filename)
        # Get 128-dimension face encoding of the loaded image - use face_recognition.face_encodings
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encodings = face_recognition.face_encodings(image)[0]

        # Append encodings and name to known_faces and known_names
        known_faces.append(encodings)
        known_names.append(name)

print(known_names)

'''
i=0
print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, known_face_locations = locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s) in {filename}')


    match = None
    for face_encoding, face_location in zip(encodings, locations):
        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding)


        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = name_to_color(match)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
           
    # Show image
    if(match== None): #Check if match is None
        personName = "unidentified" + str(i)
        cv2.imwrite(os.path.join('<path>/unidentified' , personName+'.jpg'), image)
        i+=1
    else:
        personName = match
    logSurveillance(personName)

'''


cap = cv2.VideoCapture('jordanpippen.mp4')

ret, frame = cap.read()

while cap.isOpened():
    locations = face_recognition.face_locations(frame)

    encodings = face_recognition.face_encodings(frame, known_face_locations = locations)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #############################
    


    match = None
    jordan = False
    pippen = False
    for face_encoding, face_location in zip(encodings, locations):
        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding)


        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels

            appearance = []
            for i in range(len(results)):
                if results[i] == True:
                    appearance.append(known_names[i])
            
            match = max(appearance,key=appearance.count)
            
            if match == "jordan":
                jordan_now = "True"
                if jordan_now != jordan:
                    logSurveillance("jordan")
                    jordan = True

            # Each location contains positions in order: top, right, bottom, left
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
                    color = name_to_color(match)

            # Paint frame
                    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
                    cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)


            else:
                jordan = False

            if match == "pippen":
                pippen_now = "True"
                if pippen_now != jordan:
                    logSurveillance("pippen")
                    pippen = True

            # Each location contains positions in order: top, right, bottom, left
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
                    color = name_to_color(match)

            # Paint frame
                    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
                    cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)


            else:
                pippen = False


    #############################
    cv2.imshow("recognixed faces", frame)
    ret, frame = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()