#For testing in duck game!!
from directkeys import PressKey,ReleaseKey, W, A, S, D,  J,SPACE,Q,E,SHIFT
import numpy as np
from PIL import ImageGrab
import cv2
import time
#library we created! Imports keyboard pressing
#from directkeys import PressKey,ReleaseKey, W, A, S, D,  J,SPACE,Q,E,SHIFT
from getkeys import key_check #local library
from alexnet import alexnet
import os 

WIDTH = 120
HEIGHT = 120
#Learning rate
LR = 1e-3
#Number of epochs
EPOCHS = 15
MODEL_NAME = "DuckGame-{}-{}-{}-epochs.model".format(LR,"alexnet",EPOCHS)




def draw_lines(img,lines):
    try:
        for line in lines:
            coords = line[0]
            
            linecolor = [255,255,255]
            x1y1 = (coords[0],(coords[1]))
            x2y2 = (coords[2],(coords[3]))
            lineThickness = 0.5
            #draw lines on the image
            cv2.line(img,x1y1,x2y2,linecolor,lineThickness)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img) #0's of the shape of the image
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img,mask)
    return masked



def process_img(original_image):
    #Convert image to gray to lower memory size
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #let's do edge detection with canny
    processed_img = cv2.Canny(processed_img, threshold1 = 150, threshold2 = 250)
    
    #Let's blur the lines a bit
    processed_img = cv2.GaussianBlur(processed_img, (3,3),0)
    
    #Coordinates for the region of interest. From bottem left, up and over, to bottom right coordinate.
    #vertices = np.array([[10,500],[10,300],[300,200],[500,200],[1200,300],[1200,500]])
    vertices = np.array([
            [10,screenSize[1]]   ,
            [10,int(screenSize[1]*.6)],
            [int(screenSize[0]*0.375)   ,int(screenSize[1]*.4)],
            [int(screenSize[0]*0.625) ,int(screenSize[1]*.4)],
            [screenSize[0]       ,int(screenSize[1]*.6)],
            [screenSize[0]       ,screenSize[1]]])
    #Apply the region of interest
    processed_img = roi(processed_img,[vertices])
    
    #Let's do hough lines #https://www.youtube.com/watch?v=lhMXDqQHf9g
    #Must pass edge detection to HoughLines
    lines = cv2.HoughLinesP(processed_img,
                            rho=1,
                            theta = np.pi/180,
                            threshold=180,
                            minLineLength=20,
                            maxLineGap=15)
    
    #draw lines on the original image
    draw_lines(processed_img,lines)
    
    
    return processed_img



screenSize = [1200,700]

sleeptime = .01

def up():
    PressKey(W)
    time.sleep(sleeptime)
    ReleaseKey(W)
    time.sleep(sleeptime)
    ReleaseKey(A)
    time.sleep(sleeptime)
    ReleaseKey(S)
    time.sleep(sleeptime)
    ReleaseKey(D)
    
def down():
    PressKey(S)
    time.sleep(sleeptime)
    ReleaseKey(S)
    time.sleep(sleeptime)
    ReleaseKey(W)
    time.sleep(sleeptime)
    ReleaseKey(A)
    time.sleep(sleeptime)
    ReleaseKey(D) 
    time.sleep(sleeptime)

def left():
    PressKey(A)
    time.sleep(sleeptime)
    ReleaseKey(A)
    time.sleep(sleeptime)
    ReleaseKey(S)
    time.sleep(sleeptime)
    ReleaseKey(W)
    time.sleep(sleeptime)
    ReleaseKey(D)  
    time.sleep(sleeptime)

def right():
    PressKey(D)
    time.sleep(sleeptime)
    ReleaseKey(D)
    time.sleep(sleeptime)
    ReleaseKey(S)
    time.sleep(sleeptime)
    ReleaseKey(W) 
    time.sleep(sleeptime)
    ReleaseKey(A) 
    time.sleep(sleeptime)

    
def attack():
    PressKey(J)
    time.sleep(sleeptime)
    ReleaseKey(J)

def jump():
    PressKey(SPACE)
    time.sleep(sleeptime)
    ReleaseKey(SPACE)
    print("JUMP")

def ragdoll():
    PressKey(Q)
    time.sleep(sleeptime)
    ReleaseKey(Q)
    
def pick_up_wep():
    PressKey(E)
    time.sleep(sleeptime)
    ReleaseKey(E)

def strafe():
    PressKey(SHIFT)
    time.sleep(sleeptime)
    ReleaseKey(SHIFT)


model = alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)



def main():
    #Countdown for testing keys
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    paused = False
    while(True):
        if not paused:
            screen = np.array(ImageGrab.grab(bbox=(0,40,screenSize[0],screenSize[1])))
            #screen after processing
            screen = process_img(screen)
            #cv2.imshow("window", screen)
            screen = cv2.resize(screen,(WIDTH,HEIGHT))
            
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            
            #Comment line below out to increase framerate
            #Show the original screen that is being scanned. Comment for speed.
            #cv2.imshow("window2",cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            moves = list(np.around(prediction))
            #Prints the move and the neural output
            print(moves,prediction)
            
#            if moves == [1,0,0,0,0,0,0,0,0]:
#                left()
#            elif moves == [0,1,0,0,0,0,0,0,0]:
#                up()
#            elif moves == [0,0,1,0,0,0,0,0,0]:
#                right()
#            elif moves == [0,0,0,1,0,0,0,0,0]:
#                down()
#            elif moves == [0,0,0,0,1,0,0,0,0]:
#                attack()
#            elif moves == [0,0,0,0,0,1,0,0,0]:
#                jump()
#            elif moves == [0,0,0,0,0,0,1,0,0]:
#                ragdoll()
#            elif moves == [0,0,0,0,0,0,0,1,0]:
#                pick_up_wep()
#            elif moves == [0,0,0,0,0,0,0,0,1]:
#                strafe()
            
            if moves[0] == 1:
                left()
            if moves[1] == 1:
                up()
            if moves[2] == 1:
                right()
            if moves[3] == 1:
                down()
            if moves[4] == 1:
                attack()
            if moves[5] == 1:
                jump()
            if moves[6] == 1:
                ragdoll()
            if moves[7] == 1:
                pick_up_wep()
            if moves[8] == 1:
                strafe()
            
        keys = key_check()
        
        if "T" in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(S)
                ReleaseKey(D)
                ReleaseKey(J)
                ReleaseKey(SPACE)
                ReleaseKey(Q)
                ReleaseKey(E)
                ReleaseKey(SHIFT)
                
                
        
        
        
        
main() 
        
        
        
        
        
 