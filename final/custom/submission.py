# Feel free to modifiy this file.
from torchvision import models, transforms

team_id = 5
team_name = "LXTJ"
email_address = "lj1327@nyu.edu"

def get_model():
    return models.resnet152(num_classes=800)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
