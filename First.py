import numpy as np
from PIL import ImageGrab
import cv2
import time
#library we created! Imports keyboard pressing
#from directkeys import PressKey,ReleaseKey, W, A, S, D,  J,SPACE,Q,E,SHIFT
from getkeys import key_check #local library
import os

screenSize = [1200,700]


def keys_to_output(keys):
    #[W, A, S, D,  J,SPACE,Q,E]
    output = [0,0,0,0,0,0,0,0,0]
    
    if "A" in keys:
        output[0] = 1
    if "W" in keys:
        output[1] = 1
    if "D" in keys:
        output[2] = 1
    if "S" in keys:
        output[3] = 1
    if "J" in keys:
        output[4] = 1
    if " " in keys:
        output[5] = 1
    if "Q" in keys:
        output[6] = 1
    if "E" in keys:
        output[7] = 1
    if "SHIFT" in keys:
        output[8] = 1
        
    else:
        pass
        
    return output
        


file_name = "training_data.npy"
if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name))
else:
    print("File does not exist, starting fresh")
    training_data = []

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

#create the region of interest when processing
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





def main():
    #Countdown for testing keys
    for i in list(range(1))[::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,40,screenSize[0],screenSize[1])))
        #screen after processing
        screen = process_img(screen)
        #cv2.imshow("window", screen)
        screen = cv2.resize(screen,(120,120))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        
        #Comment line below out to increase framerate

        
        
        
        
        
        #Comment line below out to increase framerate
        #Show the original screen that is being scanned. Comment for speed.
        #cv2.imshow("window2",cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        
        #if there is no remainder/if multiple of 500
        if len(training_data) % 500 == 0:

           np.save(file_name, training_data)
           print("SAVED")
           print('loop took {} seconds'.format(time.time()-last_time))
           last_time = time.time()
        
main() 
        
        
        
        
        
        
#Countdown for testing keys
#for i in list(range(4))[::-1]:
#    print(i+1)
#    time.sleep(1)
#print("Jump")
#PressKey(SPACE)
#time.sleep(.5)
#PressKey(SPACE)
#ReleaseKey(SPACE)
#print("Quack")
#PressKey(E)
#time.sleep(.5)
#ReleaseKey(E)
#time.sleep(.05)
#PressKey(E)
#time.sleep(.5)
#ReleaseKey(E)
#time.sleep(.05)
#PressKey(E)
#time.sleep(.5)
#ReleaseKey(E)


