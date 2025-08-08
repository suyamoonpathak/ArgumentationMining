from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load your image. Example: from a URL
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# --- Unconditional Captioning ---
inputs = processor(raw_image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)
print("Unconditional caption:", caption)

# --- Conditional Captioning (optional) ---
prompt = "a photography of"
inputs_cond = processor(raw_image, prompt, return_tensors="pt")
output_cond = model.generate(**inputs_cond)
caption_cond = processor.decode(output_cond[0], skip_special_tokens=True)
print("Conditional caption:", caption_cond)
