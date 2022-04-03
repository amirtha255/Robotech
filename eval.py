from PIL import Image
import numpy as np
import torchvision.transforms as T
from matplotlib import pyplot as plt

import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2
THRESHOLD = 0.8

CLASSES = ['Litter']

transform = T.Compose([
    T.ToTensor()
])

def get_instance_segmentation_model(
        num_classes, model_name='maskrcnn_resnet50_fpn'):

    model = torchvision.models.detection.__dict__[model_name](
        pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if model_name.startswith('mask') or model_name.startswith('efficientnet'):

        in_features_mask = \
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
    return model

model = get_instance_segmentation_model(NUM_CLASSES)

model.eval()

im = Image.open("/content/000006.jpg").convert('RGB')

img = transform(im).unsqueeze(0)

outputs = model(img)

plt.figure(figsize=(16,10))
img = np.array(im)[:,:,1]
plt.imshow(im)
ax = plt.gca()
keep = outputs[0]['scores'].detach().numpy() > THRESHOLD
prob = outputs[0]['scores'].detach()[keep]
labels = outputs[0]['labels'].detach().numpy()[keep].tolist()
masks = outputs[0]['masks'].detach().numpy()[keep]

if len(prob) == 0:
  max_prob = keep
else:
  max_prob = [int(np.argmax(prob))]

masking = np.zeros((1,)+img.shape)

for j, i in enumerate(outputs[0]['boxes'].detach().numpy()[max_prob].tolist()):
    p = prob[j]
    masking += masks[j]
    ax.add_patch(plt.Rectangle((i[0], i[1]), i[2] - i[0], i[3] - i[1],
                                   fill=False, color='r', linewidth=3))
    cl = int(labels[j])-1
    text = f'{CLASSES[cl]}: {p:0.2f}'
    ax.text(i[0], i[1], text, fontsize=15,
            bbox=dict(facecolor='yellow', alpha=0.5))
    
imagines = np.array(im)
imagines[:,:,0] = imagines[:,:,0] + masking[0,:,:]*100
plt.imshow(imagines)
plt.axis('off')
plt.show()

plt.figure(figsize=(16,10))
plt.imshow(masking[0,:,:])
plt.axis('off')
plt.show()
