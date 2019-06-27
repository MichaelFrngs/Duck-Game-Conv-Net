import numpy as np
from alexnet import alexnet

WIDTH = 120
HEIGHT = 120

#Learning rate
LR = 2
#Number of epochs
EPOCHS = 1000
MODEL_NAME = "DuckGame-{}-{}-{}-epochs.model".format(LR,"alexnet",EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load("training_data_v2.npy")

train = train_data[:-500]
test = train_data[-500:]

#i[0] is the image data
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_Y = [i[1] for i in test]

model.fit({"input": X} , {"targets":Y}, n_epoch = EPOCHS,
          validation_set = ({"input": test_X} , {"targets":test_Y}),
          snapshot_step = 500, #THIS IS CRASHING THE PROGRAM
          show_metric = True,
          run_id=MODEL_NAME
          )

#tensorboard --logdir=foo:C:/Users/Bunnita/Desktop/DuccBot/log

model.save(MODEL_NAME)