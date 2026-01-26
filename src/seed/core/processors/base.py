from abc import ABC, abstractmethod
from typing import Generator

from ...models import Lemma


class Processor(ABC):
    """
    Base class for processors.
    """

    @abstractmethod
    def process(
        self,
    ) -> Generator[Lemma, None, None]:
        """
        Process the input data and yield Lemma objects.

        Returns:
            Generator[Lemma, None, None]: Generator of Lemma objects.
        """
        pass
