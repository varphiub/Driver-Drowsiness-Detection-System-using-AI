from tkinter import *
from PIL import Image,ImageTk
gw = '#134e86'
g="#0a2845#"
gg ="white"
global R1
R1 = Tk()

R1.geometry('712x712')
R1.title('drowsiness detection')

R1.resizable(width = FALSE ,height= FALSE)
Image_open = Image.open("y.jpg")
image = ImageTk.PhotoImage(Image_open)
sigup = Label(R1,image=image,bg=gg)
sigup.place(x=0,y=0,bordermode="outside")

    
def head(): 
    import cv2
    import numpy as np

    

    #dinstance function
    def distance(x,y):
        import math
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2) 
        
    #capture source video
    cap = cv2.VideoCapture(0)

    #params for ShiTomasi corner detection # detects the corner points in the frame
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
    # Parameters for lucas kanade optical flow  #used to track Shi-tomasi corner points
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #TERM_CRITERIA_EPS(epsilon)-used to stop algorithm iteration if specified accuray is reached.
    #criteria =is the iteration termination criteria .when this critia is satisfied ,algorithm iteration stops.
    #TERM_CRITERIA_COUNT-
    #epsilon-is an upperbound on error of a floating point number.is the difference between 1 and largest floating point number.
    #path to face cascde
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #function to get coordinates
    def get_coords(p1):
        try: return int(p1[0][0][0]), int(p1[0][0][1])
        except: return int(p1[0][0]), int(p1[0][1])

    #define font and text color. #FONT_HERSHEY_SIMPLEX is a font type  #font types:FONT_HERSHEY_SIMPLEX,FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_SIMPLEX


    #define movement threshodls
    max_head_movement = 20
    movement_threshold = 50
    gesture_threshold = 175
    #gesture-movement of part of body or head or hand,to express an idea or meaning.

    #find the face in the image
    face_found = False
    frame_num = 0
    
    while frame_num < 30:
        # Take first frame and find corners in it
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face_found = True
        cv2.imshow('image',frame)
        #out.write(frame)
        cv2.waitKey(1)
    
    face_center = x+w/2, y+h/3
    p0 = np.array([[face_center]], np.float32)


    #gesture-movement of part of body or head or hand,to express an idea or meaning.
    gesture = False
    x_movement = 0
    y_movement = 0
    gesture_show = 60 #number of frames a gesture is shown

    while True:    
        ret,frame = cap.read()
        old_gray = frame_gray.copy()
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
        cv2.circle(frame, get_coords(p1), 4, (0,0,255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255,0,0))
        
        #get the xy coordinates for points p0 and p1
        a,b = get_coords(p0), get_coords(p1)
        #The abs() takes only one argument, a number whose absolute value is to be returned.
        #The argument can be an integer, a floating point number or a complex number.
        #If the argument is an integer or floating point number, abs() returns the absolute value in integer or float.
        x_movement += abs(a[0]-b[0])
        y_movement += abs(a[1]-b[1])
        
        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame,text,(50,50), font, 0.8,(0,0,255),2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame,text,(50,100), font, 0.8,(0,0,255),2)

        if x_movement > gesture_threshold:
            gesture = 'Not detected'
        if y_movement > gesture_threshold:
            gesture = 'Drowsiness Detected'
        if gesture and gesture_show > 0:
            cv2.putText(frame,'Status: ' + gesture,(50,50), font, 1.2,(0,0,255),3)
            gesture_show -=1
        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 60 #number of frames a gesture is shown
            
        #print distance(get_coords(p0), get_coords(p1))
        p0 = p1
        cv2.imshow('image',frame)
        #out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()



    



signUpbt = Button(R1,text = "head bend",width=10,height=2,bg='green',fg="white",font="5",relief=RAISED,overrelief=RIDGE,command=head)

signUpbt.place( x =550,y=350)
  
R1.mainloop()