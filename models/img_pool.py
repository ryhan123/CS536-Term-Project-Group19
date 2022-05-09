"""

    Essentials:
        - ImgPool is a pool of recent images
        - When we calculate our discriminator loss, we can sometimes use past images instead of most recent images
        - So we query our image pool for an image, sometimes returning a recent image, and sometimes returning a past images
"""
import random
import torch

class ImgPool():

    def __init__(self, size):

        self.size = size
        self.count = 0
        self.img_pool = []

    def query(self, images):


        if self.size == 0:
            return images

        return_images = []

        for image in images:

            image = torch.unsqueeze(image.data, 0)

            if self.count < self.size:

                self.count += 1
                self.img_pool.append(image)
                return_images.append(image)
            
            else:

                p = random.uniform(0,1)

                if p >= 0.5:

                    rand_ind = random.randint(0, self.size - 1)
                    out = self.img_pool[rand_ind].clone()
                    self.img_pool[rand_ind] = image
                    return_images.append(out)

                else:

                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)

        return return_images
