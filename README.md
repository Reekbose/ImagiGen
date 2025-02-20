# ImagiGen

# Overview 
This project focuses on classifying and naming images using machine learning techniques. It processes images, extracts features, and assigns appropriate labels based on the trained model.

# Features
 1.Loads and preprocesses images /n
 2.Extracts features for classification
 3.Applies a machine learning model for labeling
 4.Saves the results in a structured format

# How to run?
### STEPS:

Create a Python virtual environment and install Gradio using the following commands in the terminal:

```bash
pip install virtualenv 
virtualenv my_env # create a virtual environment my_env
my_env\Scripts\Activate # activate my_env
```
### STEP 01- installing required libraries in my_env

```bash
pip install langchain==0.1.11 gradio==4.44.0 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1
```

### STEP 02- Import required tools from the transformers library

In the project directory, create a Python file, Click on File Explorer, then right-click in the explorer area and select New File. Name this new file image_cap.py. copy the various code segments below and paste them into the Python file.

```bash
import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```


### Step 03: Fetch the model and initialize a tokenizer

In the next phase, we fetch an image, which will be captioned by our pre-trained model. This image can either be a local file or fetched from a URL. The Python Imaging Library, PIL, is used to open the image file and convert it into an RGB format which is suitable for the model.

```bash
img_path = "IMAGE NAME.jpeg"
image = Image.open(img_path).convert('RGB')
```

Next, the pre-processed image is passed through the processor to generate inputs in the required format. The return_tensors argument is set to "pt" to return PyTorch tensors.

```bash
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")
```

then,
```bash
outputs = model.generate(**inputs, max_length=50)
```
Finally, the generated output is a sequence of tokens. To transform these tokens into human-readable text, use the decode method provided by the processor. The skip_special_tokens argument is set to True to ignore special tokens in the output text.

```bash
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
```

Run it to see the result.
```bash
python3 image_cap.py
```

### Next,
### We use Gradio to implement image captioning app 

### STEP 01- create a new Python file and call it image_captioning_app.py

```bash
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
```

### STEP 02- Load the pretrained model

```bash
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### STEP 03- Define the image captioning function

```bash
def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    # Process the image
    inputs = processor(raw_image, return_tensors="pt")
    # Generate a caption for the image
    out = model.generate(**inputs,max_length=50)
    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
```

### STEP 04- Create the Gradio interface

```bash
iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)
```

### STEP 05- Launch the Web App

```bash
iface.launch()
```

### STEP 06-  Run the application

```bash
python image_captioning_app.py
```





