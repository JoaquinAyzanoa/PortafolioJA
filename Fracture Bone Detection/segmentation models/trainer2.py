import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self._device = 'cpu'
        if cuda:
            self._device = 'cuda:0' if t.cuda.is_available() else 'cpu'
            self._model = model.to(self._device)
            self._crit = crit.to(self._device)
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y, seg_y):
        self._optim.zero_grad()  # Reset gradients

        if self._cuda:
            x, y = x.to(self._device), y.to(self._device)
            if seg_y is not None:
                seg_y = seg_y.to(self._device)

        # Forward pass
        class_output, seg_output = self._model(x)
        
        class_loss = self._crit(class_output, y)  # Classification loss
        if seg_y is not None:
            seg_loss = self._crit(seg_output, seg_y)  # Segmentation loss
            loss = class_loss + seg_loss  # Total loss
        else:
            loss = class_loss

        # Backward pass
        loss.backward()
        self._optim.step()  # Update weights

        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        if self._cuda:
            x, y = x.to(self._device), y.to(self._device)

        # Forward pass
        outputs = self._model(x)

        # If outputs is a tuple, unpack it
        if isinstance(outputs, tuple):
            class_output, seg_output = outputs
        else:
            class_output = outputs

        # Calculate loss
        loss = self._crit(class_output, y)

        return loss.item(), class_output
        
    def train_epoch(self):
        self._model.train()
        total_loss = 0.0
        num_batches = len(self._train_dl)

        for batch in tqdm(self._train_dl, desc='Training', leave=False):
            x, y = batch[:2]  # Extract input data and classification labels

            if len(batch) == 3:
                seg_y = batch[2]  # Extract segmentation labels if available
            else:
                seg_y = None

            loss = self.train_step(x, y, seg_y)
            total_loss += loss

        return total_loss / num_batches
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc='Validation', leave=False):
                loss, preds = self.val_test_step(x, y)
                total_loss += loss
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_preds = (all_preds > 0.5) 
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return total_loss / len(self._val_test_dl), f1
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        f1_scores = []
        patience_counter = 0
        epoch = 0
        
        while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            train_loss = self.train_epoch()
            val_loss, f1 = self.val_test()

            min_val_loss = min(val_losses) if val_losses else 0
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            f1_scores.append(f1)

            print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}')

            if self._early_stopping_patience > 0:
                if val_loss < min_val_loss:
                    patience_counter = 0
                    self.save_checkpoint(epoch + 1)
                else:
                    patience_counter += 1

                if patience_counter >= self._early_stopping_patience:
                    print(f'Early stopping after {epoch + 1} epochs without improvement.')
                    break

            epoch += 1
            if epochs > 0 and epoch >= epochs:
                break

        return train_losses, val_losses, f1_scores
                    
        
        
        
