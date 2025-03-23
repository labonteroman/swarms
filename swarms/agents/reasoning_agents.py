from typing import List, Literal, Optional, Dict, Any, Union
from swarms.agents.consistency_agent import SelfConsistencyAgent
from swarms.agents.i_agent import (
    IterativeReflectiveExpansion as IREAgent,
)
from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.output_types import OutputType

# Import the HoT agent
from hcot_agent import HighlightedChainOfThoughtAgent

agent_types = Literal[
    "reasoning-duo",
    "self-consistency",
    "ire",
    "reasoning-agent",
    "consistency-agent",
    "ire-agent",
    "hot",  # Add HoT as a new agent type
    "highlighted-chain-of-thought",
]


class ReasoningAgentRouter:
    """
    A Reasoning Agent Router that selects different reasoning strategies for tasks.

    Supports multiple reasoning approaches:
    - Reasoning Duo: Two agents collaborate on a task
    - Self Consistency: Multiple reasoning paths are aggregated
    - IRE: Iterative Reflective Expansion for complex reasoning
    - HoT: Highlighted Chain of Thought with fact tagging for grounded reasoning

    Attributes:
        agent_name (str): The name of the agent.
        description (str): A brief description of the agent's capabilities.
        model_name (str): The name of the model used for reasoning.
        system_prompt (str): The prompt that guides the agent's reasoning process.
        max_loops (int): The maximum number of loops for the reasoning process.
        swarm_type (agent_types): The type of reasoning swarm to use.
        num_samples (int): The number of samples to generate for multi-path reasoning.
        output_type (OutputType): The format of the output (e.g., dict, list).
        fact_tag_style (str): Style of fact tags for HoT agent (standard, numeric, or bracket).
        temperature (float): Temperature setting for model generation.
        dataset (str): Dataset type for specialized prompt formatting.
    """

    def __init__(
            self,
            agent_name: str = "reasoning_agent",
            description: str = "A reasoning agent that can answer questions and help with tasks.",
            model_name: str = "gpt-4o",
            system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
            max_loops: int = 1,
            swarm_type: agent_types = "reasoning-duo",
            num_samples: int = 1,
            output_type: OutputType = "dict",
            fact_tag_style: str = "standard",
            temperature: float = 0.7,
            dataset: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.description = description
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_loops = max_loops
        self.swarm_type = swarm_type
        self.num_samples = num_samples
        self.output_type = output_type
        self.fact_tag_style = fact_tag_style
        self.temperature = temperature
        self.dataset = dataset

        # Keep track of the current selected agent
        self.current_agent = None

    def select_swarm(self):
        """
        Selects and initializes the appropriate reasoning swarm based on the specified swarm type.

        Returns:
            An instance of the selected reasoning swarm.
        """
        if (
                self.swarm_type == "reasoning-duo"
                or self.swarm_type == "reasoning-agent"
        ):
            self.current_agent = ReasoningDuo(
                agent_name=self.agent_name,
                agent_description=self.description,
                model_name=[self.model_name, self.model_name],
                system_prompt=self.system_prompt,
                output_type=self.output_type,
            )

        elif (
                self.swarm_type == "self-consistency"
                or self.swarm_type == "consistency-agent"
        ):
            self.current_agent = SelfConsistencyAgent(
                agent_name=self.agent_name,
                description=self.description,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                max_loops=self.max_loops,
                num_samples=self.num_samples,
                output_type=self.output_type,
            )

        elif (
                self.swarm_type == "ire" or self.swarm_type == "ire-agent"
        ):
            self.current_agent = IREAgent(
                agent_name=self.agent_name,
                description=self.description,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                max_loops=self.max_loops,
                max_iterations=self.num_samples,
                output_type=self.output_type,
            )

        elif (
                self.swarm_type == "hot"
                or self.swarm_type == "highlighted-chain-of-thought"
        ):
            self.current_agent = HighlightedChainOfThoughtAgent(
                model_name=self.model_name,
                num_samples=self.num_samples,
                temperature=self.temperature,
                fact_tag_style=self.fact_tag_style,
                use_self_consistency=(self.num_samples > 1)
            )

        else:
            raise ValueError(f"Invalid swarm type: {self.swarm_type}")

        return self.current_agent

    def run(self, task: str, *args, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Executes the selected swarm's reasoning process on the given task.

        Args:
            task (str): The task or question to be processed by the reasoning agent.
            *args, **kwargs: Additional arguments passed to the agent's run method.

        Returns:
            The result of the reasoning process.
        """
        swarm = self.select_swarm()

        # Special handling for HoT agent which may need dataset-specific prompt formatting
        if self.swarm_type in ["hot", "highlighted-chain-of-thought"] and self.dataset:
            from hcot_agent import create_hot_prompt
            formatted_task = create_hot_prompt(task, self.dataset)
            result = swarm.run(formatted_task)

            # Convert HoT output format to match expected output_type if needed
            if self.output_type == "string" and isinstance(result, dict):
                return result["final_answer"]
            return result

        return swarm.run(task=task, *args, **kwargs)

    def batched_run(self, tasks: List[str], save_path: Optional[str] = None, *args, **kwargs) -> List[Any]:
        """
        Executes the reasoning process on a batch of tasks.

        Args:
            tasks (List[str]): A list of tasks to be processed.
            save_path (Optional[str]): Path to save results (for agents that support it).
            *args, **kwargs: Additional arguments passed to the agent's run method.

        Returns:
            List of results from the reasoning process for each task.
        """
        # Special handling for HoT agent with batch processing support
        if self.swarm_type in ["hot", "highlighted-chain-of-thought"]:
            swarm = self.select_swarm()

            if self.dataset:
                # Format tasks with dataset-specific prompts
                from hcot_agent import create_hot_prompt
                formatted_tasks = [create_hot_prompt(task, self.dataset) for task in tasks]
                results = swarm.batch_run(formatted_tasks, save_path=save_path)
            else:
                results = swarm.batch_run(tasks, save_path=save_path)

            # Convert results to match expected output_type if needed
            if self.output_type == "string":
                return [r["final_answer"] if isinstance(r, dict) else r for r in results]
            return results

        # Standard processing for other agent types that don't have native batch support
        results = []
        for task in tasks:
            results.append(self.run(task, *args, **kwargs))
        return results

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Returns information about the currently selected agent.

        Returns:
            Dict containing agent type, configuration, and capabilities.
        """
        return {
            "agent_type": self.swarm_type,
            "model": self.model_name,
            "samples": self.num_samples,
            "output_type": self.output_type,
            "supports_batching": self.swarm_type in ["hot", "highlighted-chain-of-thought"],
            "supports_fact_tagging": self.swarm_type in ["hot", "highlighted-chain-of-thought"],
        }