#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

class JustificationGenerator:
    """
    This class generates a justification using OpenAI models, based on a given question and answer.

    Args:
        model_type (str): 3 types of model: text-davinci-003, gpt-3.5-turbo, gpt-4. For gpt-3.5-turbo and gpt-4, the prompt formats are same.
        system (str): A system message used in gpt-3.5-turbo model and gpt-4 (default vs. customized).
        definition (str): A string that contains descriptions/ definitions of the biomedical terms.
        question (str): A question to be answered.
        justification_prompt (str): A prompt used to query the OpenAI model for justifications.
    """

    def __init__(self, model_type: str, system: str,
                 definition: str, question: str,
                 justification_prompt: str):

        # Verify input parameters
        if model_type not in ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4']:
            raise ValueError("model_type must be one of ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4']")

        if not all(isinstance(i, str) for i in [system, definition, question, justification_prompt]):
            raise TypeError("system, definition, question, and justification_prompt must be string.")

        # Initialize class attributes
        self.definition = definition
        self.question = question
        self.system = system
        self.model_type = model_type
        self.justification_prompt = justification_prompt


    def __str__(self):
        """ Returns a string representation of the JustificationGenerator instance. """
        return f"Initiated JustificationGenerator Instance:\nModel: {self.model_type}\nSystem Message: {self.system}\nBio Definitions: {self.definition}\nQuestion: {self.question}\nJustification Prompt: {self.justification_prompt}"

    def get_prompt(self, context: str, review: str):
        """
        Generate a prompt.

        Args:
            context (str): A string that provides the context for the prompt.
            review (str): Answer to the question.

        Returns:
        str or list of dict: The formatted prompt based on the model type.
        """
        # Generate the base prompt
        context_prompt = f"{self.definition}\n{context}\nQuestion: {self.question}\nAnswer: {review}.\n{self.justification_prompt} "

        if self.model_type == 'text-davinci-003':
            return context_prompt

        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4']:
            return [
                {"role": "system", "content": f"{self.system}"},
                {"role": "user", "content": f"{context_prompt}"}
            ]






