from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories = ('black', 'grizzly', 'teddy')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

title = "Bear Classifier"
description = " Upload a picture of a bear or drag and drop one of the examples below to the upload box to find out what type of bear it is!\n" + "This model has been trained to identify a black bear, a grizzly bear and a teddy bear."

image = gr.Image(shape=(192,192))
label = gr.Label()
examples = ['grizzly.jpg', 'black.jpg', 'teddy.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples = examples, title=title, description=description)
intf.launch(inline=False)