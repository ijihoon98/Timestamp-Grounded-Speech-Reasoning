import asyncio
import os
import random
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, AsyncORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()

class MultipleChoiceORM(ORM):

    def __call__(self, completions, label, **kwargs) -> List[float]:
        """
        Evaluates completions based on whether the last line contains the correct answer label.

        Args:
            completions (list[str]): Generated outputs
            label (list[str]): Ground truth labels (e.g., "a", "b", "c", "d")

        Returns:
            list[float]: Reward scores (1.0 if correct, 0.0 otherwise)
        """
        rewards = []
        for completion, gt_label in zip(completions, label):
            try:
                gt_label_lower = gt_label.lower().strip()

                # Get the last non-empty line
                lines = [line.strip() for line in completion.strip().split('\n') if line.strip()]
                last_line = lines[-1].lower() if lines else ''

                # Find (a), (b), (c), (d) pattern in the last line
                match = re.search(r'\(([a-d])\)', last_line)
                if match:
                    predicted = match.group(1)
                    if predicted == gt_label_lower:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    # No valid answer found
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards


orms['multiple_choice'] = MultipleChoiceORM


class TimestampCompactionORM(ORM):
    """
    TimestampCompactionORM.
    - Max reward: 0.5 (instead of 1.0)
    - Min reward: 0.1 (constant for timestamps >= max_count, instead of 0.0)
    """

    def __init__(self, optimal_count: int = 1, max_count: int = 5, require_timestamp: bool = True):
        """
        Args:
            optimal_count: The ideal number of timestamp chunks (default: 1)
            max_count: Number of timestamps at which reward becomes minimum (default: 5)
            require_timestamp: If True, responses with 0 timestamps get 0 reward (default: True)
        """
        self.optimal_count = optimal_count
        self.max_count = max_count
        self.require_timestamp = require_timestamp
        self.max_reward = 0.5
        self.min_reward = 0.1

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Evaluates completions based on timestamp compaction.

        Reward scheme:
            - 0 timestamps: 0.0 (if require_timestamp=True) or 0.5 (if False)
            - 1 timestamp (optimal): 0.5
            - 2-4 timestamps: linear decay from 0.5 to 0.1
            - 5+ timestamps: 0.1 (constant)

        Args:
            completions (list[str]): Generated outputs

        Returns:
            list[float]: Reward scores based on timestamp compaction
        """
        timestamp_patterns = [
            r'starts?\s+at\s+\d+\.?\d*\s*(?:seconds?|s)?\s+and\s+ends?\s+at\s+\d+\.?\d*\s*(?:seconds?|s)?',
            r'(?:is\s+)?from\s+\d+\.?\d*\s*(?:seconds?|s)?\s*to\s+\d+\.?\d*\s*(?:seconds?|s)?',
            r'\(\d+\.?\d*s?\s*[-–]\s*\d+\.?\d*s?\)',
            r'\(\d+\.?\d*\s*,\s*\d+\.?\d*\)',
            r'\d+\.?\d*\s*(?:to|[-–])\s*\d+\.?\d*\s*seconds?',
        ]

        rewards = []
        for completion in completions:
            timestamp_count = 0
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, completion, re.IGNORECASE)
                timestamp_count += len(matches)

            if timestamp_count == 0:
                reward = 0.0 if self.require_timestamp else self.max_reward
            elif timestamp_count <= self.optimal_count:
                reward = self.max_reward
            elif timestamp_count >= self.max_count:
                reward = self.min_reward
            else:
                # Linear decay between optimal and max
                reward = self.max_reward - (timestamp_count - self.optimal_count) * \
                    (self.max_reward - self.min_reward) / (self.max_count - self.optimal_count)

            rewards.append(reward)

        return rewards


orms['timestamp_compaction'] = TimestampCompactionORM

