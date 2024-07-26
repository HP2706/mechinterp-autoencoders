from anthropic import Anthropic, AsyncAnthropic
from instructor import Instructor, AsyncInstructor
import asyncio
import instructor
from typing import (
    Any, AsyncGenerator, TypeAlias, overload, TypeVar, Awaitable,
    List, Optional, Type, TypeVar, Any, Coroutine, 
    Literal, Union, cast
)

from openai import AsyncOpenAI, OpenAI
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

T = TypeVar('T')

MaybeAwaitable = Union[T, Awaitable[T]]


class AutomatedInterpretability:
    def __init__(
        self, 
        client : Union[OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic],
        model : str = 'gpt-4-turbo',
        use_logfire : bool = True,
        mode = instructor.Mode.TOOLS
    ) -> None:
        if isinstance(client, (Anthropic, AsyncAnthropic)):
            self.client = instructor.from_anthropic(client, mode=mode)
        elif isinstance(client, (OpenAI, AsyncOpenAI)):
            self.client = instructor.from_openai(client, mode=mode)
            if use_logfire:
                import logfire
                logfire.instrument_openai(client)
        self.model = model
        
    
    def client_is_async(self) -> bool:
        return isinstance(self.client, AsyncInstructor)

    def check_client(self, async_mode: bool = False):
        if async_mode and not self.client_is_async():
            raise ValueError("Client is not async")
        if not async_mode and self.client_is_async():
            raise ValueError("Client is async")

    async def aggregate_explanation_async(
        self,
        examples: List[FeatureSample], 
        max_samples_per_prompt: int = 2,
        feature_or_neuron: Literal["feature", "neuron"] = "feature",
        image_provider: Literal['openai', 'anthropic', 'gemini'] = 'openai',
        max_retries: int = 2
    ) -> Union[ActivationHypothesis, InconclusiveHypothesis]:
        self.check_client(async_mode=True)     
        hypothesis = []
        for i in range(0, len(examples), max_samples_per_prompt):
            hypothesis.append(
                self.explain_activation_async(
                    examples[i:i+max_samples_per_prompt], 
                    feature_or_neuron,
                    image_provider,
                    max_retries
                )
            )

        #we wait for all the hypothesis to be generated before summarizing them
        hypothesis = await asyncio.gather(*hypothesis) 
            
        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text", 
                    "text": f"""
                        You are a machine learning scientist.
                        You previously came up with a list of hypothesis for why the {feature_or_neuron} is active 
                        based on different non overlapping examples.
                        Your job is to aggregate these hypothesis into one hypothesis. 
                        or alternatively return an inconclusive hypothesis if the activation hypothesis do not have 
                        an overlapping theme/explanation.
                        """
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": '\n'.join(hyp.stringify() for hyp in hypothesis)
                }]
            }
        ]

        return cast(Union[ActivationHypothesis, InconclusiveHypothesis], await self.client.chat.completions.create(
            response_model=Union[ActivationHypothesis, InconclusiveHypothesis], #type: ignore
            model=self.model,
            max_retries=max_retries,
            messages=messages #type: ignore
        ))

    def aggregate_explanation_sync(
        self,
        examples: List[FeatureSample], 
        max_samples_per_prompt: int = 2,
        feature_or_neuron: Literal["feature", "neuron"] = "feature",
        image_provider: Literal['openai', 'anthropic', 'gemini'] = 'openai',
        max_retries: int = 2
    ) -> Union[ActivationHypothesis, InconclusiveHypothesis]:
        self.check_client(async_mode=False)
        
         
        """ hypothesis = []
        for i in range(0, len(examples), max_samples_per_prompt):
            hypothesis.append(
                self.explain_activation_async(
                    examples[i:i+max_samples_per_prompt], 
                    feature_or_neuron,
                    image_provider,
                    max_retries
                )
            ) """

        hypothesis = [
            ActivationHypothesis
            (
                attributes="",
                hypothesis="The feature is active when it sees a banana",
                conviction=5,
                reasoning="The feature is active when it sees a banana"
            ),
        ]

        messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": f"""
                        You are a machine learning scientist.
                        You previously came up with a list of hypothesis for why the {feature_or_neuron} is active 
                        based on different non overlapping examples.
                        Your job is to aggregate these hypothesis into one hypothesis. 
                        or alternatively return an inconclusive hypothesis if the activation hypothesis do not have 
                        an overlapping theme/explanation.
                        """
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": '\n'.join(hyp.stringify() for hyp in hypothesis)
                }
            ]
        }]
        
        return cast(Union[ActivationHypothesis, InconclusiveHypothesis], self.client.chat.create(
            response_model=Union[ActivationHypothesis, InconclusiveHypothesis],
            model=self.model,
            max_retries=max_retries,
            messages=messages
        ))
       
    async def explain_activation_async(
        self,
        examples : List[FeatureSample], 
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
        image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai',
        max_retries: int = 2
    ) -> Union[ActivationHypothesis, InconclusiveHypothesis]:
        self.check_client(async_mode=True)
        return await self.explain_activation_core( #type: ignore
            examples, feature_or_neuron, image_provider, max_retries, async_mode=True
        )
    
    def explain_activation_sync(
        self,
        examples : List[FeatureSample], 
        feature_or_neuron : Literal["feature", "neuron"] = "feature",
        image_provider : Literal['openai', 'anthropic', 'gemini'] = 'openai',
        max_retries: int = 2
    ) -> Union[ActivationHypothesis, InconclusiveHypothesis]:
        self.check_client(async_mode=False)
        return cast(Union[ActivationHypothesis, InconclusiveHypothesis], self.explain_activation_core(
            examples, feature_or_neuron, image_provider, max_retries, async_mode=False
        ))

    def explain_activation_core(
        self,
        examples : List[FeatureSample], 
        feature_or_neuron : Literal["feature", "neuron"],
        image_provider : Literal['openai', 'anthropic', 'gemini'],
        max_retries: int,
        async_mode: bool
    ) -> MaybeAwaitable[Union[ActivationHypothesis, InconclusiveHypothesis]]:
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

        messages = [
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
                        Note you must return in the format: 
                        '{{"content" : {{"the json format of either ActivationHypothesis or InconclusiveHypothesis"}}}}'
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

        if async_mode:
            return self.client.chat.completions.create(
                response_model=Union[ActivationHypothesis, InconclusiveHypothesis],
                model=self.model,
                max_retries=max_retries,
                messages=messages #type: ignore
            )
        else:
            return self.client.chat.create(
                response_model=Union[ActivationHypothesis, InconclusiveHypothesis],
                model=self.model,
                max_retries=max_retries,
                messages=messages #type: ignore
            )
    
    # Public methods that use the core function
    def predict_activation_sync(self,
        unseen_examples: List[FeatureSample],
        hypothesis: ActivationHypothesis,
        max_tries: int = 2,
        feature_or_neuron: Literal["feature", "neuron"] = "feature"
    ) -> Optional[List[PredictActivation]]: 
        #this will ever if this synchronous function is run with asyncio.run() 
        # as 2 event loops will be created
        
        self.check_client(async_mode=False)

        loop = asyncio.get_event_loop()
        coroutine = self.predict_activation_core(
            unseen_examples, 
            hypothesis, 
            max_tries, 
            feature_or_neuron, 
            async_mode=False
        )
        return cast(
            Optional[List[PredictActivation]], 
            loop.run_until_complete(coroutine)
        )
        

    async def predict_activation_async(self,
        unseen_examples: List[FeatureSample],
        hypothesis: ActivationHypothesis,
        max_tries: int = 2,
        feature_or_neuron: Literal["feature", "neuron"] = "feature"
    ) -> Awaitable[Optional[List[PredictActivation]]]:
        self.check_client(async_mode=True)
        return cast(
            Awaitable[Optional[List[PredictActivation]]], 
            self.predict_activation_core(
                unseen_examples, 
                hypothesis, 
                max_tries, 
                feature_or_neuron, 
                async_mode=True
        ))

    async def predict_activation_core(
        self,
        unseen_examples: List[FeatureSample],
        hypothesis: ActivationHypothesis,
        max_tries: int,
        feature_or_neuron: Literal["feature", "neuron"],
        async_mode: bool,
    ) -> MaybeAwaitable[Optional[List[PredictActivation]]]:
        """
        Core function to predict activations, handling both synchronous and asynchronous execution.
        """

        def create_message() -> List[dict]:
            return [
                {
                    "role": "system",
                    "content": f"""
                        You are a machine learning scientist.
                        Your job is to predict the activation of a {feature_or_neuron}.
                        You previously came up with the following hypothesis for when the {feature_or_neuron} is active:
                        hypothesis: {hypothesis.hypothesis}
                        degree of conviction: {hypothesis.conviction}
                        reasoning: {hypothesis.reasoning}
                        with this in mind, predict the quantized activation (an integer) caused by the following context:
                        {formatted_examples}
                        return in the specified json format.
                        You need to predict the activation for each example.
                        Return as many predictions as there are examples.
                        {err_msg if err_msg else ""}
                    """
                }
            ]

        formatted_examples = '\n'.join(str(elm) for elm in unseen_examples)
        err_msg = None

        for _ in range(max_tries):
            try:
                messages = create_message()
                if async_mode:
                    response = await self.client.chat.completions.create( 
                        response_model=List[PredictActivation],
                        model=self.model,
                        messages=messages #type: ignore
                    )
                else:
                    response = cast(List[PredictActivation], self.client.chat.completions.create(
                        response_model=List[PredictActivation],
                        model=self.model,
                        messages=messages #type: ignore
                    ))

                if len(response) != len(unseen_examples):
                    raise ValueError(f"Expected {len(unseen_examples)} predictions, got {len(response)}")
                return response
            except Exception as e:
                err_msg = f"Error: {e}"
                if _ == max_tries - 1:
                    return None

    
    def predict_next_logit_sync(
        self,
        examples : List[FeatureSample],
        hypothesis : ActivationHypothesis
    ) -> PredictNextLogit:
        return cast(PredictNextLogit, self.predict_next_logit_core(
            examples, hypothesis
        ))
    
    async def predict_next_logit_async(
        self,
        examples : List[FeatureSample],
        hypothesis : ActivationHypothesis
    ) -> Awaitable[PredictNextLogit]:
        return cast(Awaitable[PredictNextLogit], self.predict_next_logit_core(
            examples, hypothesis
        ))


    def predict_next_logit_core(
        self,
        examples : List[FeatureSample],
        hypothesis : ActivationHypothesis,
    ) -> MaybeAwaitable[PredictNextLogit]:
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
