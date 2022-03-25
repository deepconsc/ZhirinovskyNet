import torch 
from torch import nn 
from torchvision import models
from torch.nn.functional import softmax as Softmax 

class ZhirinovskyNet(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = models.resnet50(
            pretrained=True
            ).eval().to(self.device)

    @torch.inference_mode(mode=True)
    def forward(self, input_tensor: torch.tensor):
        """
        Forward module.

        Receives:
            input_tensor -> torch.tensor 
                preprocessed tensor of shape 1xCxWxH

        Returns:
            Probability of 341th ImageNet class. 
            Derived from: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a, 
            it's a "Squealer", who seemed to serve as Minister of Propaganda
            from George Orwell's 1945's novel. 
        """
        outputs = self.encoder(input_tensor.to(self.device))
        outputs = Softmax(outputs, dim=0)
        return outputs[0][341].item()
        