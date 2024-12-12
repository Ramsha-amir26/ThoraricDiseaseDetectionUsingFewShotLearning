import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt 
from tqdm import tqdm_notebook
from tqdm import tnrange

from utilities import MEAN, STD


# create samples
def extract_sample(n_way, n_support, n_query, datax, datay):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task to sample
      n_support (int): number of images per class in the support set
      n_query (int): number of images per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
      
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
        (NumPy array): class_labels (labels of the randomly selected classes)
  """
  sample = []

  # from Total No. of classes in an array of data (np.unique(datay)), randomly select n_way classes
  K = np.random.choice(np.unique(datay), n_way, replace=False) # returns a numpy array with a size equal to n_way

  for cls in K: # cls = data class

    datax_cls = datax[datay == cls] # get the images corresponding to our class

    perm = np.random.permutation(datax_cls) # randomly shuffle the images of this class

    sample_cls = perm[:(n_support+n_query)] #  select a sample, which is == (no. of support images + no. of query images)

    sample.append(sample_cls) # add sample images to list, we end up with [[images of cls_1],[images of cls_2],..., [images of class cls_n_way]]

  sample = np.array(sample) # convert list to numpy array

  sample = torch.from_numpy(sample) # convert to a tensor

  sample = sample.type(torch.float32) / 255.0 # convert to float and scale the image data to [0, 1]

  # The above sample has dimensions [n_way, n_support+n_query, img_height, img_width, channels]
  # re-arange the dimensions so that channels are first
  sample = sample.permute(4, 2, 3, 0, 1) # [channels, img_height, img_width, n_way, n_support+n_query]

  # standardize the data by subtracting the mean and dividing by the standard deviation
  sample = (sample - MEAN[:,None, None, None, None]) / STD[:, None, None, None, None]

  # rearange the dimensions to the original ones [n_way, n_support+n_query, img_height, img_width, channels]
  sample = sample.permute(3, 4, 1, 2, 0)

  # Since we are using PyTorch, input into our model should be of the form  [n_way, n_support+n_query, channels, img_height, img_width]
  sample = sample.permute(0,1,4,2,3)
  
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'class_labels': K
      })


def display_sample(sample):
  """
  Displays sample in a grid

  Args:
      sample (torch.Tensor): sample of images to display      
  """
  #need 4D tensor to create grid, currently 5D
  sample_4D = sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])  

  #make a grid
  out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
  plt.figure(figsize=(16,7))

  # out has dimension [channels, img_height, img_width]

  out = out.permute(1, 2, 0) # [img_height, img_width, channels]
  out = out * STD[None, None, :] + MEAN[None, None, :] # remember you are only multiplying by the channels dimension

  plt.imshow(out)


# Prediction on a sample of data
def predict(model, sample, device="cpu"):
  """
    Args:
      model (object): trained prototypical model
      sample (dict): dictionary containing the following keys:
                                                              images - images for the support + query set
                                                              n_way - number of classes to sample
                                                              n_support - number of support images
                                                              n_query - number of query images 
      device (str): device to run the model on - 'cpu' or 'cuda'

    Returns:
      output (dict): dictionary with the following keys:
                                                        loss - loss value
                                                        acc - accuracy of prediction
                                                        y_hat = prediction tensor for each query image in each class
  """
  model.to(device)
  l, output = model.set_forward_loss(sample)

  return output
