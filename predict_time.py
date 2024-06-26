import torch
import torch.nn as nn
from numpy import array as nparray
# include whatever other imports you need here

class TimePredictionNetwork(nn.Module):
   # Your network definition goes here
   def __init__(self):
      super(TimePredictionNetwork,self).__init__()
      self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 16 x 224 x 224
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 32 x 112 x 112
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 64 x 56 x 56
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4), # 128 x 14 x 14
            nn.Flatten(),
            nn.Dropout(0.5)
        )

      self.MLPhours = nn.Sequential(
         nn.Linear(in_features=128*14*14,out_features=128),
         nn.ReLU(),
         nn.Linear(in_features=128,out_features=12)
      )

      self.MLPmins = nn.Sequential(
         nn.Linear(in_features=128*14*14,out_features=64),
         nn.ReLU(),
         nn.Linear(in_features=64,out_features=1)
      )


   def forward(self,x):
      x = self.layers(x)
      x = x.view(x.size(0),-1)
      hours = self.MLPhours(x)
      mins = self.MLPmins(x)
      return hours,mins.view(-1)

def predict(images):
   # Determine which device the input tensor is on
   device = torch.device("cuda" if images.is_cuda else "cpu")

   model = TimePredictionNetwork() # Add your model init parameters here if you have any
   # Move to same device as input images
   model = model.to(device)
   # Load network weights
   model.load_state_dict(torch.load('weights.pkl',map_location=torch.device(device)))
   # Put model in evaluation mode
   model.eval()

   # Optional: do whatever preprocessing you do on the images
   # if not included as tranformations inside the model

   with torch.no_grad():
      # Pass images to model
      predicted_times = model(images)

   # If your output needs any post-processing, do it here
   predictions = []
   for i in range(len(images)):
      time = []
      mins_pred = divmod(int(round(predicted_times[1][i].item())),60)
      time.append(int(predicted_times[0][i].argmax().item()+mins_pred[0]))
      time.append(round(mins_pred[1]))
      predictions.append(time)
   predictions = nparray(predictions)
   predicted_times = torch.from_numpy(predictions)

   # Return predicted times
   return predicted_times