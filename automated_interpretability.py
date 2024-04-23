from instructor import Instructor, AsyncInstructor
from typing import Any, AsyncGenerator, List, Optional, Type, TypeVar, Any, Coroutine, Literal
from datamodels import PredictActivation, ActivationHypothesis, PredictNextLogit, ActivationExample
    
async def explain_activation(
    client : AsyncInstructor,
    model : str, 
    examples : List[ActivationExample], 
    feature_or_neuron : Literal["feature", "neuron"]
 ) -> Coroutine[Any, Any, ActivationHypothesis]:
    '''predicts an explanation based on earlier examples
    Args:
        client : AsyncInstructor, the client to use to predict activations
        model : str, the model to predict activations for
        examples : TODO,
        feature_or_neuron : Literal["feature", "neuron"], whether to predict a feature or neuron
    '''
    return client.chat.create(
        response_model=ActivationHypothesis,
        model=model,
        messages=[
            {
                "role": "system", 
                "content":f"""
                    You are a machine learning scientist.
                    You job is to create an explanation of a {feature_or_neuron} based on emperical observations 
                    of in what contexts the {feature_or_neuron} is active and to what degree.
                    You will base your answer on the following examples, where you will see the context, 
                    the token and the associated activation for that token:
                    {examples}
                    return in the specified json format
                """
            }
        ]
    )


async def predict_activation(
    client : AsyncInstructor,
    model : str, 
    examples : List[ActivationExample], 
    hypothesis : ActivationHypothesis,
    feature_or_neuron : Literal["feature", "neuron"]
) -> Coroutine[Any, Any, PredictActivation]:
    '''predicts activations for either a feature or neuron based on 9 token examples
    Args:
        client : AsyncInstructor, the client to use to predict activations
        model : str, the model to predict activations for
        examples : str, 9 token examples to predict activations for
    '''
    examples_stringified = '\n'.join(example.model_dump_json() for example in examples)
    return client.chat.create(
        response_model=PredictActivation,
        model=model,
        messages=[
            {
                "role": "system", 
                "content":f"""
                    You are a machine learning scientist.
                    You job is to predict the activation of a {feature_or_neuron}. 
                    You previously came up with the following hypothesis for when the {feature_or_neuron} is active:
                    {hypothesis.hypothesis}
                    with this in mind, predict the activation caused by the following context:
                    {examples_stringified}
                    return in the specified json format
                """
            }
        ]
    )

async def predict_next_logit(
    model : str, 
    client : AsyncInstructor,
    examples : List[ActivationExample],
    hypothesis : ActivationHypothesis
) -> Coroutine[Any, Any, PredictNextLogit]:
    '''
    Using the explanations of features generated in the previous 
    analysis, we ask a language model to predict if a previously 
    unseen logit token is something the feature should predict as 
    likely to come next
    '''
    
    examples_stringified = '\n'.join(example.model_dump_json() for example in examples)
    return client.chat.create(
        response_model=PredictNextLogit,
        model=model,
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


async def interpret_activations(model : str, examples ):
    '''
        given n examples the model returns a plausible hypothesis 
        for when the feature or neuron is active
    '''