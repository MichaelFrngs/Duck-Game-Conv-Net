import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os 

os.chdir("C:/Users/Bunnita/Desktop/DuccBot")

train_data = np.load('training_data.npy')

#LOOK AT THE DATA
#for data in train_data:
#    img = data[0]
#    choice = data[1]
#    cv2.imshow("test",img)
#    print("A, W, D, S, J,SPACE,Q,E,SHIFT")
#    print(choice)
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break


df = pd.DataFrame(train_data)
#print(df.head)
print(Counter(df[1].apply(str)))
len(Counter(df[1].apply(str)))

As = []
Ws = []
Ds = []
Ss = []
Js = []
SPACEs = []
Qs = []
Es = []
SHIFTs = [] 

shuffle(train_data)


for data in train_data:
    img = data[0]
    choice = data[1]
    print(choice)
    if choice[0] == 1:
        As.append([img,choice])
       # print("A")
    if choice[1] == 1:
        Ws.append([img,choice])
       # print("W")
    if choice[2] == 1:
        Ds.append([img,choice])
       # print("D")
    if choice[3] == 1:
        Ss.append([img,choice])
      #  print("S")
    if choice[4] == 1:
        Js.append([img,choice])
    if choice[5] == 1:
        SPACEs.append([img,choice])
        #print("SPACE")
    if choice[6] == 1:
        Qs.append([img,choice])
        print("Q")
        print("Q")
    if choice[7] == 1:
        Es.append([img,choice])
    if choice[8] == 1:
        SHIFTs.append([img,choice])
        print("SHIFT")


#Balance the data
As	=	As[:len(Ds)][:len(Ss)][:len(SPACEs)]
Ws	=	Ws[:len(As)]
Ds	=	Ds[:len(As)]
Ss	=	Ss[:len(As)]
Js	=	Js[:len(As)]
SPACEs	=	SPACEs[:len(As)]
Qs	=	Qs[:len(As)]
Es	=	Es[:len(As)]
SHIFTs	=	SHIFTs[:len(As)]


    
    
    
    
for i in train_data:
    #image
    #print(i[0])
    #predictions
    print(i[1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    if choice == [1,0,0,0,0,0,0,0,0]:
#        As.append([img,choice])
#    elif choice == [0,1,0,0,0,0,0,0,0]:
#        Ws.append([img,choice])
#    elif choice == [0,0,1,0,0,0,0,0,0]:
#        Ds.append([img,choice])
#    elif choice == [0,0,0,1,0,0,0,0,0]:
#        Ss.append([img,choice])
#    elif choice == [0,0,0,0,1,0,0,0,0]:
#        Js.append([img,choice])
#    elif choice == [0,0,0,0,0,1,0,0,0]:
#        SPACEs.append([img,choice])
#    elif choice == [0,0,0,0,0,0,1,0,0]:
#        Qs.append([img,choice])
#    elif choice == [0,0,0,0,0,0,0,1,0]:
#        Es.append([img,choice])
#    elif choice == [0,0,0,0,0,0,0,0,1]:
#        SHIFTs.append([img,choice])
#    else:
#        print("no match!")
            
final_data = As+Ws+Ds+Ss+Js+SPACEs+Qs+Es+SHIFTs
#print("FINAL DATA",final_data[1])
#print("Training",train_data[1])

shuffle(final_data)
print(len(final_data))

for i in final_data:
    print(i[1])
np.save("training_data_v2.npy",final_data)
