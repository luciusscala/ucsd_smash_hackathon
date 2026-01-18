from PIL import Image
import numpy as np
import tensorflow as tf
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("imageomics/sentinel-beetles")
ds['train'].add_column('tensors', np.zeros(len(ds['train'])))

def resize(image):
    
    img = image
    width, height = img.size

    new_h = 256
    new_w = int(width * (new_h / height))  
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    final_size = 256
    padded = Image.new(
        mode=resized.mode,
        size=(final_size, final_size),
        color=(255, 255, 255) if resized.mode == 'RGB' else 255
    )

    x_offset = (final_size - resized.width) // 2
    y_offset = 0 
    padded.paste(resized, (x_offset, y_offset))

    #convert padded + resized image to numpy array
    img_array = np.array(padded)

    #convert to tensor
    tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    tensor = tensor / 255.0

    return tensor


for i in range(len(ds['train']['file_path'])):
    print(ds['train']['file_path'][i])
    new_tensor = resize(ds['train']['file_path'][i])
    ds['train'][i]['tensors'] = new_tensor
    print(i)

ds.save_to_disk("./data/dataset_w_tensors")





