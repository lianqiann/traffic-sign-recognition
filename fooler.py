import torch
import torchvision.transforms as transforms
import numpy as np

import cv2
from matplotlib import pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

cuda_available = torch.cuda.is_available()

def prepare_image(image_cv2, do_normalize=True):
      # Resize
      img = cv2.resize(image_cv2, (128, 128))
      img = img[:, :, ::-1].copy()
      # Convert to tensor
      tensor_img = transforms.functional.to_tensor(img)

      # Possibly normalize
      if do_normalize:
         tensor_img = normalize(tensor_img)
      # Put image in a batch
      batch_tensor_img = torch.unsqueeze(tensor_img, 0)

      # Put the image in the gpu
      if cuda_available:
        batch_tensor_img = batch_tensor_img.cuda()
      return batch_tensor_img


def UnNormalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]):
      std_arr = torch.tensor(std)[:, None, None]
      mean_arr = torch.tensor(mean)[:, None, None]
      def func(img):
        img = img.clone()
        img *= std_arr
        img += mean_arr
        return img
      return func
unnormalize = UnNormalize()

def obtain_image(tensor_img, do_normalize=True):
      tensor_img = tensor_img.cpu()
      if do_normalize:
        tensor_img = unnormalize(tensor_img).detach().numpy()
      img = tensor_img.transpose(1,2,0)
      #img = transforms.functional.to_pil_image((tensor_img.data))
      return img
def top_classes(values, sign_classes, top_k=5):
    sorted_classes = np.argsort(-values.detach().numpy())
    class_ids = sorted_classes[:top_k]
    class_names = [sign_classes[it] for it in list(class_ids)]
    
    return class_names



class StepImage():
      def __init__(self, orig_input, step_size=2, is_normalized=True, 
                   renorm=True, eps=30, norm_update='l2'):
        self.orig_input = orig_input
        if is_normalized:
          mean=[0.485, 0.456, 0.406]
          std= [0.229, 0.224, 0.225]
        else:
          mean=[0., 0., 0.]
          std= [1., 1., 1.]

        is_cuda = orig_input.is_cuda
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
        if is_cuda:
          self.mean = self.mean.cuda()
          self.std = self.std.cuda()
        self.eps = eps
        self.renorm = renorm
        self.step_size = step_size
        self.norm_update = norm_update
    
      def project(self, x):
        """
        """
        diff = x - self.orig_input
        if self.renorm:
          diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        val_projected = self.orig_input + diff

        val_projected *= self.std
        val_projected += self.mean
        val_clamped = torch.clamp(val_projected, 0, 1)
        val_clamped -= self.mean
        val_clamped /= self.std
        return val_clamped
  
      def step(self, x, g):
        step_size = self.step_size
        # Scale g so that each element of the batch is at least norm 1
        if self.norm_update == 'l2':
              l = len(x.shape) - 1
              g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        else:
              g_norm = torch.torch.abs(g).mean()
        scaled_g = g / (g_norm + 1e-10)
        stepped = x + scaled_g * step_size
        projected = self.project(stepped)
        return projected


def get_fooler_image(args, model, sign_classes, device):

    
        # We set it in eval, so that batch normalization layers are not updated
        model.eval()
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            model.cuda()


        # Load the image we will be playing with
        img = cv2.imread(args.PATH+f'/examples/{args.image_name}')
        
        # Prepare the image
        batch_normalized_img = prepare_image(img)
        # Run it through the network
        output = model(batch_normalized_img.to(device))
        output_probs = torch.nn.functional.log_softmax(output).squeeze(0)

        class_names = top_classes(output_probs, sign_classes)
        #get probability of top prediction
        print('the top5 predictions of original image are:', class_names)
        print('the probability of the top predictions is :', np.exp(output_probs.data.cpu().detach().numpy().max()))

        # generate fooler
        starting_image = prepare_image(img)
        batch_tensor = starting_image.clone().requires_grad_(True)
        step = StepImage(starting_image, step_size=0.05, 
                        is_normalized=True, renorm=False, norm_update='abs')
        for _ in range(args.n_epochs):
            inputs = batch_tensor.to(device)
            objective = torch.nn.functional.log_softmax(model.forward(inputs),1)[0,args.fooler_class]
            gradient, = torch.autograd.grad(objective, inputs)
            batch_tensor = step.step(batch_tensor, gradient)

        
        output = model(batch_tensor)
        output_probs = torch.nn.functional.log_softmax(output).squeeze(0)
        class_names = top_classes(output_probs, sign_classes)
        
        print('the top5 predictions of fooler are:', class_names)
        print('the probability of the top predictions is :', np.exp(output_probs.data.cpu().detach().numpy().max()))
        

        image = obtain_image(batch_tensor[0, :], do_normalize=True)
    
        
        plt.imsave(args.PATH+f'outputs/fooled{args.fooler_class}_{args.image_name}', image)
