import os
from PIL import Image

# Directory path where the images are located
directory = '/home/roboken03/Downloads/Data'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image file
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)

        x,y = image.size
        min = x if x < y else y
        image = image.crop(((x - min) / 2, (y - min) / 2, (x + min) / 2, (y + min) / 2))
        # Resize the image to 640x640 pixels
        resized_image = image.resize((640, 640))

        # Save the resized image
        resized_image.save(image_path)

        # Close the image file
        image.close()