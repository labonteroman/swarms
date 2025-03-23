import os
import re
import random
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
import pandas as pd
from loguru import logger

# Agent system prompts
HOT_SYSTEM_PROMPT = """
You are a reasoning agent that uses Highlighted Chain of Thought (HoT). When answering questions, you'll first identify key phrases from the question and tag them with <fact1>, <fact2>, etc. Then, in your explanation, you'll reference these facts using the same tags to ground your reasoning. This makes your thought process transparent and traceable. After your explanation, provide a final answer in curly brackets {}.
"""

REFORMULATION_SYSTEM_PROMPT = """
You are a question reformulation agent. Your task is to identify the key phrases in a question that are most relevant to answering it, and tag them with <fact1>, <fact2>, etc. Focus on facts, entities, and details that will be necessary for answering the question correctly. Do not alter the meaning of the question.
"""

REASONING_SYSTEM_PROMPT = """
You are a reasoning agent that explains your thought process step by step. When constructing your explanation, reference the facts from the reformulated question using their fact tags (<fact1>, <fact2>, etc). Make sure your reasoning is grounded in the information provided in the question. After your explanation, provide your final answer in curly brackets {}.
"""

AGGREGATION_SYSTEM_PROMPT = """
You are an aggregation agent. You will be given multiple responses to the same question, each containing a final answer in curly brackets {}. Your task is to:
1. Analyze all responses
2. Identify the most common or well-supported answer
3. Provide a single final answer with a brief explanation of why you chose it

If answers differ, explain your reasoning for selecting one over the others.
"""


class Agent:
    """Base agent class"""

    def __init__(
            self,
            agent_name: str,
            description: str,
            model_name: str = "gpt-4o",
            system_prompt: str = None,
            temperature: float = 0.7,
            max_tokens: int = 2048
    ):
        self.agent_name = agent_name
        self.description = description
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, task: str) -> str:
        """Run the agent on a task"""
        # This would be implemented with an actual API call
        # For now, we'll just return a placeholder
        return f"Response from {self.agent_name} using {self.model_name} on task: {task[:30]}..."


class ReformulationAgent(Agent):
    """Agent that identifies and tags key facts in a question"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.5):
        super().__init__(
            agent_name="Reformulation-Agent",
            description="An agent that identifies and tags key facts in a question",
            model_name=model_name,
            system_prompt=REFORMULATION_SYSTEM_PROMPT,
            temperature=temperature
        )

    def run(self, question: str) -> str:
        """Tag important facts in the question"""
        # This would call the LLM API with appropriate prompting
        # For demonstration, we'll return a formatted response
        return f"Reformatted Question: {question}"  # In real implementation, this would include fact tags


class ReasoningAgent(Agent):
    """Agent that generates explanations grounded in tagged facts"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.7):
        super().__init__(
            agent_name="Reasoning-Agent",
            description="An agent that generates explanations grounded in tagged facts",
            model_name=model_name,
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=temperature
        )

    def run(self, reformulated_question: str) -> str:
        """Generate an explanation grounded in the tagged facts"""
        # This would call the LLM API with appropriate prompting
        # For demonstration, we'll return a formatted response
        return f"Answer: Based on <fact1> and <fact2>, I conclude that... {{Final answer}}"


class AggregationAgent(Agent):
    """Agent that aggregates multiple responses"""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        super().__init__(
            agent_name="Aggregation-Agent",
            description="An agent that aggregates multiple responses into a single final answer",
            model_name=model_name,
            system_prompt=AGGREGATION_SYSTEM_PROMPT,
            temperature=temperature
        )

    def run(self, responses: List[str]) -> str:
        """Aggregate multiple responses into a single answer"""
        task = "\n\n".join([f"Response {i + 1}: {r}" for i, r in enumerate(responses)])
        # This would call the LLM API with appropriate prompting
        # For demonstration, we'll simulate simple majority voting
        return f"Final aggregated answer based on {len(responses)} responses: {{Consensus answer}}"


class HighlightedChainOfThoughtAgent:
    """
    Highlighted Chain of Thought (HoT) agent that uses multiple sub-agents to:
    1. Reformulate questions with tagged facts
    2. Generate grounded explanations referencing those facts
    3. Aggregate multiple reasoning paths for self-consistency
    """

    def __init__(
            self,
            model_name: str = "gpt-4o",
            num_samples: int = 3,
            temperature: float = 0.7,
            fact_tag_style: str = "standard",  # "standard", "numeric", or "bracket"
            use_self_consistency: bool = True
    ):
        self.model_name = model_name
        self.num_samples = num_samples
        self.temperature = temperature
        self.fact_tag_style = fact_tag_style
        self.use_self_consistency = use_self_consistency

        # Initialize sub-agents
        self.reformulation_agent = ReformulationAgent(model_name=model_name, temperature=temperature)
        self.reasoning_agent = ReasoningAgent(model_name=model_name, temperature=temperature)
        self.aggregation_agent = AggregationAgent(model_name=model_name,
                                                  temperature=temperature * 0.5)  # Lower temperature for aggregation

        # Set up logging
        logger.info(f"Initialized HoT Agent with {model_name}, {num_samples} samples, temp={temperature}")

    def _format_fact_tags(self, text: str) -> str:
        """Format fact tags based on the selected style"""
        if self.fact_tag_style == "numeric":
            # Replace <fact1> with <1>, etc.
            return re.sub(r'<fact(\d+)>', r'<\1>', text)
        elif self.fact_tag_style == "bracket":
            # Replace <fact1> with [1], etc.
            return re.sub(r'<fact(\d+)>', r'[\1]', text)
        else:
            # Standard <fact1> format
            return text

    def run_single_path(self, question: str) -> str:
        """Run a single reasoning path through the pipeline"""
        # Step 1: Reformulate question with fact tags
        reformulated_question = self.reformulation_agent.run(question)
        reformulated_question = self._format_fact_tags(reformulated_question)

        # Step 2: Generate grounded explanation and answer
        response = self.reasoning_agent.run(reformulated_question)

        return response

    def run(self, question: str) -> Dict:
        """Run the complete HoT pipeline with optional self-consistency"""
        result = {
            "question": question,
            "reformulated_question": None,
            "responses": [],
            "final_answer": None
        }

        if self.use_self_consistency:
            # Generate multiple reasoning paths in parallel
            responses = []

            logger.info(f"Generating {self.num_samples} responses concurrently...")
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.run_single_path, question): i
                    for i in range(self.num_samples)
                }
                for future in as_completed(futures):
                    response = future.result()
                    responses.append(response)

            result["responses"] = responses

            # Aggregate the responses
            final_answer = self.aggregation_agent.run(responses)
            result["final_answer"] = final_answer
        else:
            # Just run a single path
            response = self.run_single_path(question)
            result["responses"] = [response]
            result["final_answer"] = response

        return result

    def batch_run(self, questions: List[str], save_path: Optional[str] = None) -> List[Dict]:
        """Run the HoT pipeline on a batch of questions"""
        results = []

        for i, question in tqdm(enumerate(questions), total=len(questions)):
            logger.info(f"Processing question {i + 1}/{len(questions)}")
            result = self.run(question)
            results.append(result)

        if save_path:
            # Save results to CSV
            data = []
            for i, result in enumerate(results):
                data.append({
                    "id": i,
                    "question": result["question"],
                    "final_answer": result["final_answer"]
                })
            df = pd.DataFrame(data)
            df.to_csv(save_path, index=False)
            logger.info(f"Results saved to {save_path}")

        return results


class APIHandler:
    """Handler for API calls to different LLM providers"""

    @staticmethod
    def call_api(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
        """
        Call the appropriate API based on the model name

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3", "gemini-pro")
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response text or None if the request failed
        """
        # This would implement actual API calls
        # For now, we return a placeholder
        counter = 0
        max_retries = 3

        while counter < max_retries:
            try:
                # Simulate API call
                return f"API response from {model} with temp={temperature}"
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                counter += 1
                if counter == max_retries:
                    return None

        return None


def create_hot_prompt(question: str, dataset: str) -> str:
    """
    Create a prompt for the HoT agent based on the dataset and question
    Similar to the create_prompt function in main.py
    """
    # Extract the last sentence for instruction customization based on dataset
    if dataset == 'commonsenseQA':
        last_sentence_pattern = re.compile(r"Question:\s*(.*?)\s*([^.?!]*[.?!])\s*Answer Choices:", re.DOTALL)
        match = last_sentence_pattern.search(question)
        if match:
            last_sentence = match.group(2)
        else:
            last_sentence = 'the question'
    elif dataset == 'sports':
        last_sentence = 'Is the following sentence plausible?'
    elif dataset == 'reclor':
        last_sentence = 'Choose the correct answer.'
    else:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', question)
        last_sentence = sentences[-1].strip() if sentences else question

    # Build HoT instruction
    instruction = f"""I want you to answer this question but your explanation should contain references referring back to the information in the question.

To do that:
1. First, re-generate the question with proper tags (<fact1>, <fact2>, <fact3>, etc.) for key phrases that are most relevant to answering {last_sentence}
2. Then, generate your answer, making sure to reference the tagged facts in your explanation
3. Finally, provide your definitive answer in curly brackets {{}}

The output format should be:

Reformatted Question: [question with fact tags]
Answer: [explanation with fact references] {{final answer}}
"""

    prompt = f"{question}\n\n{instruction}"
    return prompt


def main():
    """Example usage of the HoT agent"""
    # Initialize the agent
    hot_agent = HighlightedChainOfThoughtAgent(
        model_name="gpt-4o",
        num_samples=3,
        temperature=0.7,
        use_self_consistency=True
    )

    # Example question
    question = "The heart pumps blood through the body. What is the main function of red blood cells in this process?"

    # Run the agent
    result = hot_agent.run(question)

    # Print the result
    print("\nQuestion:")
    print(question)
    print("\nFinal Answer:")
    print(result["final_answer"])

    # Batch processing example
    questions = [
        "The heart pumps blood through the body. What is the main function of red blood cells in this process?",
        "If a train travels at 60 miles per hour for 3 hours, how far does it travel?",
        "Water boils at 100 degrees Celsius at sea level. What happens to the boiling point as altitude increases?"
    ]

    results = hot_agent.batch_run(questions, save_path="hot_results.csv")


if __name__ == "__main__":
    main()