import torch
from torch import nn
from criteria.clip_loss import CLIPLoss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        #! Jun 18: the "inputs" is the dict of results from the model
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               
class DirClipLoss(nn.Module):
    """wrapper for clip loss"""
    def __init__(self, device):
        super(DirClipLoss, self).__init__()
        self.loss = CLIPLoss(device)

    def forward(self, src_img, src_text, results, target_text):
        #! Jun 18: the "inputs" is the dict of results from the model
        target_img = results['rgb_fine']
        target_img = torch.squeeze(0)
        target_img = torch.view(50,50,3) #todo remove magic size
        src_img = torch.squeeze(0)
        src_img = torch.view(50,50,3) #todo remove magic size
        loss = self.loss.clip_directional_loss(src_img, src_text, target_img, target_text)
        if 'rgb_coarse' in results:
            target_img_coarse = results['rgb_coarse']
            target_img_coarse = torch.squeeze(0)
            target_img_coarse = torch.view(50,50,3) #todo remove magic size
            loss += self.loss.clip_directional_loss(src_img, src_text, target_img_coarse, target_text)
        return loss

loss_dict = {'mse': MSELoss,
            'dirClip':DirClipLoss}