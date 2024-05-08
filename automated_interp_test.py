from typing import Awaitable, Literal, Union
from automated_interpretability import AutomatedInterpretability
import instructor
from openai import OpenAI, AsyncOpenAI
from datamodels import ActivationHypothesis, FeatureDescription, FeatureSample, ImageContent, TextContent, InconclusiveHypothesis, save_html

Models = Literal['gpt-4-turbo', 'claude-3-opus-20240229', 'gemini/gemini-1.5-pro-latest']
async def test_vision_prompt(model :Models):
    mapping : dict[Models, str]= {
        'gpt-4-turbo': 'openai',
        'claude-3-opus-20240229': 'anthropic',
        'gemini/gemini-1.5-pro-latest': 'openai'
    }

    #the model should not have high conviction on this examples, it should not be able to make a good observation
    bad_vision_examples = [
        FeatureSample(
            activation=0.0,
            content=ImageContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg",
                caption="an ant"
            ),
            quantized_activation=10
        ),
        FeatureSample(
            content=ImageContent(
                image_url="https://t4.ftcdn.net/jpg/01/87/47/59/360_F_187475954_KuDIRQZbGTwwyMTXZeRmojQH9YeSjrWt.jpg",
                caption="an ocean"
            ),
            activation=0.0,
            quantized_activation=10
        ),
        FeatureSample(
            activation=0.0,
            content=ImageContent(
                image_url="https://images.ctfassets.net/1aemqu6a6t65/5rPsNLkgpwvZPSmjhE5ChB/9ee0978b3d792515d59fefab5296b832/wall-street-photo-tagger-yancey-iv-nyc-and-company-02-2?w=1200&h=800&q=75",
                caption="a photo of wall street"
            ),
            quantized_activation=10
        ),
        FeatureSample(
            activation=0.0,
            content=ImageContent(
                image_url="https://t4.ftcdn.net/jpg/03/41/35/87/360_F_341358792_J5F4PkZh1Qwy8rrWiHUHEPCK6hL4AXOK.jpg",
                caption="a photo of the bull on wall street"
            ),
            quantized_activation=5
        ),
        FeatureSample(
            activation=0.0,
            content=ImageContent(
                image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/2324px-Banana-Single.jpg",
                caption="a banana"
            ),
            quantized_activation=5
        ),
        FeatureSample(
            activation=0.0,
            content=ImageContent(
                image_url="https://media.tacdn.com/media/attractions-content--1x-1/0b/a5/f8/ab.jpg",
                caption="a place in the us"
            ),
            quantized_activation=3
        ),
    ] 

    from instructor.retry import InstructorRetryException
    save_html(bad_vision_examples, 'examples.html')
    pipeline = AutomatedInterpretability(AsyncOpenAI(), model) 
    try:
        explanation =  await pipeline.aggregate_explanation_async(
            examples = [
                FeatureSample(
                    activation=0.0,
                    content=TextContent( #type: ignore
                        text="an ant",
                        token="ant",
                        token_id=1,
                        positions=[1]
                    ),
                    quantized_activation=10
                )
            ], 
            image_provider=mapping[model] #type: ignore
        ) #type: ignore
    except InstructorRetryException as e:
        print("retry error", e)
        raise Exception("InstructorRetryException should not be raised")
    
    if not isinstance(explanation, InconclusiveHypothesis):
        raise Exception("Explanation is not an InconclusiveHypothesis, it should be.")
