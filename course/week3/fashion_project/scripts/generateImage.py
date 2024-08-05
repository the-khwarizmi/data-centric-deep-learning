import torch
from torchvision import transforms

prod_data = torch.load('./data/production/dataset.pt')
prod_images = prod_data['images']

for i in range(20):
    prod_image = prod_images[i]  # vary to see a few

    # Save an image to disk
    prod_image = transforms.ToPILImage()(prod_image)
    prod_image.save('./images/test'+str(i)+'.png') # take a look at this