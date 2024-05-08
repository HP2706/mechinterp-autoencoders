from anthropic import Anthropic
from instructor import Instructor, AsyncInstructor
import instructor
from typing import Any, AsyncGenerator, List, Optional, Type, TypeVar, Any, Coroutine, Literal, Union

from openai import OpenAI
from datamodels import InconclusiveHypothesis, PredictActivation, ActivationHypothesis, PredictNextLogit, FeatureSample, ImageContent
from PIL import Image
import numpy as np
#image computability metrics 

def brightness(img: Image.Image) -> float:
    '''returns the brightness of an image'''
    return np.mean(np.array(img))

def color_diversity(img: Image.Image) -> int:
    """Calculate the number of unique colors in an image.
    
    Args:
        img (Image.Image): The image to calculate color diversity for.

    Returns:
        int: The number of unique colors.
    """
    img = img.convert('RGB')  # ensure image is in RGB format
    array = np.asarray(img)
    unique_colors = np.unique(array.reshape(-1, array.shape[2]), axis=0)
    return len(unique_colors)

class AutomatedInterpretability:
    def __init__(
        self, 
        client : Union[OpenAI, Anthropic],
        model : str = 'gpt-4-turbo',
        use_logfire : bool = True
    ) -> None:
        if isinstance(client, Anthropic):
            self.client = instructor.from_anthropic(client)
        elif isinstance(client, OpenAI):
            self.client = instructor.from_openai(client)
            if use_logfire:
                import logfire
                logfire.instrument_openai(client)
        self.model = model
        

    def aggregate_explanation(
        self,
        examples : List[FeatureSample], 
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
        image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai',
        max_samples_per_prompt : int = 5
    ):
        '''it prompts for an explanation for a set of examples, the prompts and llm to merge the explanations'''
        pass
        
        hypothesis : List[Union[ActivationHypothesis, InconclusiveHypothesis]] = []
        for i in range(0, len(examples), max_samples_per_prompt):
            hypothesis.append(
                self.explain_activation(
                    examples[i:i+max_samples_per_prompt], 
                    feature_or_neuron,
                    image_provider
                )
            )

        return self.client.chat.create(
            response_model=Union[ActivationHypothesis, InconclusiveHypothesis],
            model=self.model,
            max_retries=1,
            messages=[
                {
                    "role": "system", #type: ignore
                    "content": [ 
                        { #type: ignore
                            "type": "text", 
                            "text": f"""
                            You are a machine learning scientist.
                            You previously came up with a list of hypothesis for why the {feature_or_neuron} is active 
                            based on different non overlapping examples.
                            Your job is to aggregate these hypothesis into one hypothesis. 
                            or alternatively return an inconclusive hypothesis if the activation hypothesis do not have 
                            an overlapping theme/explanation.
                            """
                        },
                    ]
                },
                {
                    "role": "user", #type: ignore
                    "content": [
                        '\n'.join(
                            hyp.stringify() for hyp in hypothesis
                        ) 
                    ]
                }
            ]
        )

    def explain_activation(
        self,
        examples : List[FeatureSample], 
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
        image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai'
    ) -> Union[ActivationHypothesis, InconclusiveHypothesis]:
        '''predicts an explanation based on earlier examples
        Args:
            client : AsyncInstructor, the client to use to predict activations
            model : str, the model to predict activations for
            examples : List[FeatureSample], the examples of different activations, and the immediate context in which they appear
            feature_or_neuron : Literal["feature", "neuron"], whether to predict a feature or neuron
        '''
        assert all(isinstance(elm, FeatureSample) for elm in examples), "All examples should be LaionRowData or dict"

        formatted_examples = []        
        for elm in examples:
            formatted_examples.extend(
                elm.format_for_api(image_provider)
            )

        return self.client.chat.create(
            response_model=Union[ActivationHypothesis, InconclusiveHypothesis],
            model=self.model,
            max_retries=1,
            messages=[
                {
                    "role": "system", #type: ignore
                    "content": [ 
                        { #type: ignore
                            "type": "text", 
                            "text": f"""
                            You are a machine learning scientist.
                            You job is to analyze the following examples, and if posqible, find an hypothesis for 
                            why the {feature_or_neuron} is active in the given context. In many cases this is not possible,
                            and you should return a json in format InconclusiveHypothesis. Do not hesitate to return this if
                            you are not able to find an explanation. With that said, abstract explanation as long as evidence supports it.
                            For each example you will see the data(text, video, images, audio) and a quantized activation of the {feature_or_neuron} is given in the interval [0, 9] 
                            the higher the more active the {feature_or_neuron} is.
                            the conviction should be a number between 1 and 5. Be nuanced.
                            return in the specified json format. 
                            """
                        },
                    ]
                },
                {
                    "role": "user", #type: ignore
                    "content": [
                        *formatted_examples 
                    ]
                }
            ]
        )

    def predict_activation(
        self,
        unseen_examples : List[FeatureSample], 
        hypothesis : ActivationHypothesis,
        max_tries : int = 2,
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
    ) -> Optional[List[PredictActivation]]:
        '''
        from anthropic_paper:
        
        predicts activations for either a feature or neuron based on 9 token unseen_examples
        Args:
            client : AsyncInstructor, the client to use to predict activations
            model : str, the model to predict activations for
            unseen_examples : str, 9 token unseen_examples to predict activations for
        Returns:
            List[PredictActivation], the predicted activations for the unseen_examples
        '''

        formatted_examples = '\n'.join(
            f"{elm}" for elm in unseen_examples
        )
        
        err_msg : Optional[str] = None
        while max_tries > 0:
            try:
                out =  self.client.chat.create(
                    response_model=List[PredictActivation],
                    validation_context={"data": unseen_examples},
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content":f"""
                                You are a machine learning scientist.
                                You job is to predict the activation of a {feature_or_neuron}. 
                                You previously came up with the following hypothesis for when the {feature_or_neuron} is active:
                                hypothesis: {hypothesis.hypothesis}
                                degree of conviction: {hypothesis.conviction} 
                                reasoning: {hypothesis.reasoning}
                                with this in mind, predict the quantized activation(an integer) caused by the following context:
                                {formatted_examples}
                                return in the specified json format. 
                                You need to predict the activation for each example. 
                                Return as many predictions as there are examples. 
                                {err_msg if err_msg else ""}
                            """
                        }
                    ]
                )
                if len(out) != len(unseen_examples):
                    raise ValueError(f"Expected {len(unseen_examples)} predictions, got {len(out)}")
                return out
            
            except Exception as e:
                print(f"Error: {e}")
                max_tries -= 1
                if max_tries == 0:
                    return None
                else:
                    if err_msg:
                        err_msg += f"\nError: {e}"
                    else:
                        err_msg = f"Error: {e}"
                    continue
        return []

    def predict_next_logit(
        self,
        examples : List[FeatureSample],
        hypothesis : ActivationHypothesis
    ) -> PredictNextLogit:
        '''
        Using the explanations of features generated in the previous 
        analysis, we ask a language model to predict if a previously 
        unseen logit token is something the feature should predict as 
        likely to come next
        '''
        
        examples_stringified = '\n'.join(example.model_dump_json() for example in examples)
        return self.client.chat.create(
            response_model=PredictNextLogit,
            model=self.model,
            max_retries=2,
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                        You are a machine learning scientist.
                        You study the behaviour of features or neurons in an ml model.
                        You have previously come up with the following hypothesis for 
                        what the feature does {hypothesis.hypothesis} with a conviction of {hypothesis.conviction} 
                        and reasoning {hypothesis.reasoning}:
                        Your job is to, given the following examples, predict whether the 
                        Now you observe the following examples:
                        {examples_stringified}
                    """
                },
                {   "role": "user", 
                    "content": "What is the next token likely to be?"
                }
            ]
        )
