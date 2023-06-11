import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_dense_adj
import numpy as np
'''
      rand_train_test_idx function:
            * Inputs:   
                  - label: torch tensor of labels
                  - train_prop: proportion of data to be used for training
                  - valid_prop: proportion of data to be used for validation
                  - ignore_negative: boolean, if True, ignores negative labels
                  - seed: random seed
            * Outputs:
                  - train_mask: torch tensor of boolean values, True if data is in training set
                  - val_mask: torch tensor of boolean values, True if data is in validation set
                  - test_mask: torch tensor of boolean values, True if data is in test set
            * Description:
                  - randomly splits label into train/valid/test splits
'''
def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True,seed=1234):
      
      """ randomly splits label into train/valid/test splits """
      train_idx, test_idx = train_test_split(np.arange(len(label)), train_size=train_prop, random_state=seed)
      val_idx, test_idx = train_test_split(test_idx, train_size=train_prop, random_state=seed)
      train_mask = torch.zeros(len(label), dtype=torch.bool)
      train_mask[train_idx] = True
      val_mask = torch.zeros(len(label), dtype=torch.bool)
      val_mask[val_idx] = True
      test_mask = torch.zeros(len(label), dtype=torch.bool)
      test_mask[test_idx] = True
      return train_mask, val_mask, test_mask
'''
      even_quantile_labels function:
            * Inputs:
                  - vals: np array of values
                  - nclasses: number of classes to split vals into
                  - verbose: boolean, if True, prints out class label intervals
            * Outputs:
                  - label: np array of int class labels
            * Description:
                  - partitions vals into nclasses by a quantile based split,
                  where the first class is less than the 1/nclasses quantile,
                  second class is less than the 2/nclasses quantile, and so on
'''
def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = np.ones(vals.shape[0]) * -1
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
          print('Class Label Intervals:')
          for class_idx, interval in enumerate(interval_lst):
                print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label
'''
      train_adj function:
            * Inputs:
                  - adj: torch tensor of adjacency matrix
                  - data: torch tensor of data
                  - model: torch model
                  - optimizer: torch optimizer
            * Outputs:
                  - loss: torch tensor of loss
                  - new_adj: torch tensor of new adjacency matrix
            * Description:
                  - performs a single forward pass only tacking into account the pump loss
'''
def train_adj(adj,data,model,optimizer):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      _,loss,new_adj = model(data.x, adj)  # Perform a single forward pass.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss,new_adj

'''
      train function:
            * Inputs:
                  - adj: torch tensor of adjacency matrix
                  - data: torch tensor of data
                  - model: torch model
                  - train_mask: torch tensor of boolean values, True if data is in training set
                  - optimizer: torch optimizer
                  - criterion: torch loss function
            * Outputs:
                  - loss: torch tensor of loss
                  - train_acc: float of training accuracy
            * Description:
                  - performs the training step of the model, taking into account the pump loss 
                  and the cross entropy loss, only tacking into account the training nodes.
'''
def train(adj,data,model,train_mask,optimizer,criterion):
      model.train() # Set model to training mode.
      optimizer.zero_grad()  # Clear gradients.
      out,loss_norm = model(data.x, adj)  # Perform a single forward pass.
      #Get the accuracy of the model
      pred = out.argmax(dim=1).squeeze(0)  # Use the class with highest probability.
      train_correct = pred[train_mask] == data.y[train_mask]  # Check against ground-truth labels.
      train_acc = int(train_correct.sum()) / int(train_mask.sum())  # Derive ratio of correct predictions.
      loss =  loss_norm + criterion(out[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss,train_acc
  
'''
      val function:
            * Inputs:
                  - adj: torch tensor of adjacency matrix
                  - data: torch tensor of data
                  - model: torch model
                  - val_mask: torch tensor of boolean values, True if data is in validation set 
            * Outputs:
                  - val_acc: float of validation accuracy
            * Description:
                  - performs the validation step of the model, only tacking into account the validation nodes.
'''  
def val(adj,data,model,val_mask):
      model.eval() # Set model to evaluation mode.
      out,_ = model(data.x, adj) # Perform a single forward pass.
      pred = out.argmax(dim=1).squeeze(0)  # Use the class with highest probability.
      test_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
      return test_acc
'''
      test function:
            * Inputs:
                  - adj: torch tensor of adjacency matrix
                  - data: torch tensor of data
                  - model: torch model
                  - test_mask: torch tensor of boolean values, True if data is in validation set 
            * Outputs:
                  - test_acc: float of validation accuracy
            * Description:
                  - performs the validation step of the model, only tacking into account the testing nodes.
'''    
def test(adj,data,model,test_mask):
      model.eval() # Set model to evaluation mode.
      out,_ = model(data.x, adj) # Perform a single forward pass.
      pred = out.argmax(dim=1).squeeze(0)  # Use the class with highest probability.
      test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc
