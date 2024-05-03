from instructor import Instructor, AsyncInstructor
from typing import Any, AsyncGenerator, List, Optional, Type, TypeVar, Any, Coroutine, Literal, Union
from datamodels import PredictActivation, ActivationHypothesis, PredictNextLogit, ActivationExample, LaionRowData
from utils import format_image_anthropic, format_image_openai, remove_keys
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
        client : Instructor,
        model : str = 'gpt-4-turbo'
    ) -> None:
        self.client = client
        self.model = model
        
    def explain_activation(
        self,
        examples : Union[List[LaionRowData], List[ActivationExample]], 
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
        image_provider : Literal['openai', 'anthropic'] = 'openai'
    ) -> ActivationHypothesis:
        '''predicts an explanation based on earlier examples
        Args:
            client : AsyncInstructor, the client to use to predict activations
            model : str, the model to predict activations for
            examples : List[ActivationExample], the examples of different activations, and the immediate context in which they appear
            feature_or_neuron : Literal["feature", "neuron"], whether to predict a feature or neuron
        '''
        assert all (isinstance(elm, LaionRowData) for elm in examples) or all (isinstance(elm, ActivationExample) for elm in examples), "All examples should be LaionRowData or dict"

        formatted_examples = []

        
        for elm in examples:
            if isinstance(elm, LaionRowData):
                formatted_examples.extend(
                    [
                        { #type: ignore
                            "type": "text", 
                            "text": f"""
                                caption:{elm.caption}\n
                                quantized_activation:{elm.quantized_activation}
                                for below image
                            """

                        },
                        format_image_openai(elm.image_url) if image_provider == 'openai' else format_image_anthropic(elm.image_url)
                    ]
                )
            else:
                formatted_examples.extend(
                    [
                        { #type: ignore
                            "type": "text", 
                            "text": f"{elm.model_dump()}\nThe image is below" #
                        },
                    ]
                )
    
        print("formatted_examples", formatted_examples)

        return self.client.chat.create(
            response_model=ActivationHypothesis,
            model=self.model,
            messages=[
                {
                    "role": "system", #type: ignore
                    "content": [ 
                        { #type: ignore
                            "type": "text", 
                            "text": f"""
                            You are a machine learning scientist.
                            You job is to create an explanation of a {feature_or_neuron} based on emperical observations 
                            of in what contexts the {feature_or_neuron} is active and to what degree.
                            You will base your answer on the following examples, where you will see the context, 
                            the token and the associated activation for that token.
                            for each example a quantized activation of the {feature_or_neuron} is given in the interval [0, 9] 
                            the higher the more active the {feature_or_neuron} is.
                            return in the specified json format
                            """
                        },
                    ]
                },
                {
                    "role": "user", #type: ignore
                    "content": [
                        #{'type': 'text', 'text': 'the images:'},
                        *formatted_examples 
                    ]
                }
            ]
        )

    def predict_activation(
        self,
        unseen_examples : List[dict], 
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
                                {hypothesis.hypothesis}
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
        examples : List[ActivationExample],
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
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                        You are a machine learning scientist.
                        You study the behaviour of features or neurons in an ml model.
                        You have previously come up with the following hypothesis for 
                        what the feature does {hypothesis.hypothesis}:
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
