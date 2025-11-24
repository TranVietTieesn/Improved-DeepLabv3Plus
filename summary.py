#--------------------------------------------#
#   This code is used to view network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from torchviz import make_dot
from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 3
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))

    # Create random input
    dummy_input = torch.randn(3, 3, input_shape[0], input_shape[1]).to(device)

    # Calculate model output
    output = model(dummy_input)

    # Use torchviz to create visualization graph
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render('model_graph', format='png')  # Save visualization graph as PNG format

  # dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
   # flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #a = make_dot(model(1, 3, 64, 64))
    #a.view()
    #print(model)
    #--------------------------------------------------------#
    #   flops * 2 is because profile doesn't count convolution as two operations
    #   Some papers count convolution as two operations: multiplication and addition. Multiply by 2 in this case
    #   Some papers only consider multiplication operations, ignoring addition. Don't multiply by 2 in this case
    #   This code chooses to multiply by 2, following YOLOX.
    #--------------------------------------------------------#
   # flops           = flops * 2
    #flops, params   = clever_format([flops, params], "%.3f")
    #print('Total GFLOPS: %s' % (flops))
    #print('Total params: %s' % (params))
