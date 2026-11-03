import json
import logging
import re
from statistics import fmean
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import boto3
from botocore.config import Config

from mesa_types import Document
from utils.aws import AWS

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_pricing import PricingClient


class TokenPrices(NamedTuple):
    per_input_token: float
    per_output_token: float


class CostEstimator:
    """Upper-bound USD cost estimator for generating samples via Bedrock batch inference.

    Args:
        model (str): Bedrock model or inference profile id the samples are
            generated with
        price_list_model (str): Name of the model in the AWS price list
            (e.g. `Claude Sonnet 4.5`)
        region (str): AWS region the batch runs in
        system_prompt (str): System prompt sent with every document
        max_tokens (int): Maximum output tokens per sample, all of which are
            assumed to be used

    """

    __UNIT_MULTIPLIERS: ClassVar[dict[str, int]] = {"": 1, "K": 1_000, "M": 1_000_000}
    __TOKEN_COUNT_SAMPLE_SIZE: ClassVar[int] = 50

    def __init__(
        self,
        model: str,
        price_list_model: str,
        region: str,
        system_prompt: str,
        max_tokens: int,
    ) -> None:
        self.__logger: logging.Logger = logging.getLogger(__name__)
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())

        self.__model: str = model
        self.__base_model: str = model.removeprefix("eu.")
        self.__price_list_model: str = price_list_model
        self.__region: str = region
        self.__system_prompt: str = system_prompt
        self.__max_tokens: int = max_tokens
        self.__estimated_sample_size: int = 0
        self._runtime: "BedrockRuntimeClient" = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"mode": "adaptive"}),
        )
        self._pricing: "PricingClient" = boto3.client(
            "pricing", region_name="us-east-1"
        )

    def _prices(self) -> TokenPrices:
        prices: dict[str, float] = {
            product["product"]["attributes"]["inferenceType"]: (
                CostEstimator.__price_per_token(
                    next(
                        iter(
                            next(iter(product["terms"]["OnDemand"].values()))[
                                "priceDimensions"
                            ].values()
                        )
                    )
                )
            )
            for product in map(
                json.loads,
                self._pricing.get_products(
                    ServiceCode="AmazonBedrockFoundationModels",
                    Filters=[
                        {
                            "Type": "TERM_MATCH",
                            "Field": "model",
                            "Value": self.__price_list_model,
                        },
                        {
                            "Type": "TERM_MATCH",
                            "Field": "regionCode",
                            "Value": self.__region,
                        },
                        {
                            "Type": "TERM_MATCH",
                            "Field": "feature",
                            "Value": "Batch Inference",
                        },
                    ],
                )["PriceList"],
            )
            if "_Global" not in product["product"]["attributes"]["usagetype"]
        }
        if not {"Input tokens", "Output tokens"} <= prices.keys():
            raise ValueError(
                f"No batch token prices for {self.__price_list_model} in {self.__region}"
            )
        return TokenPrices(prices["Input tokens"], prices["Output tokens"])

    @staticmethod
    def __price_per_token(dimension: dict[str, Any]) -> float:
        if not (unit := re.fullmatch(r"(\d+)([KM]?) tokens", dimension["unit"])):
            raise ValueError(f"Unrecognised price unit: {dimension['unit']}")
        return float(dimension["pricePerUnit"]["USD"]) / (
            int(unit[1]) * CostEstimator.__UNIT_MULTIPLIERS[unit[2]]
        )

    def _count_tokens(self, document: Document) -> int:
        return self._runtime.count_tokens(
            modelId=self.__base_model,
            input={
                "invokeModel": {
                    "body": json.dumps(
                        AWS.create_anthropic_bedrock_batch_entry(
                            "",
                            self.__system_prompt,
                            document.content,
                            self.__max_tokens,
                        )["modelInput"]
                    )
                }
            },
        )["inputTokens"]

    def estimate(self, documents: list[Document], sample_size: int) -> float:
        """Estimate the cost of generating a sample size.

        Args:
            documents (list[Document]): Documents in the order they would be
                sent for generation
            sample_size (int): Number of samples to estimate the cost of,
                drawn from the start of `documents`

        Returns:
            float: The upper-bound USD cost of the sample size

        """
        prices: TokenPrices = self._prices()
        run: list[Document] = documents[:sample_size]
        counted: int = min(len(run), CostEstimator.__TOKEN_COUNT_SAMPLE_SIZE)
        token_counts: list[int] = [
            self._count_tokens(run[index * len(run) // counted])
            for index in range(counted)
        ]
        cost: float = (
            len(run)
            * (
                fmean(token_counts) * prices.per_input_token
                + self.__max_tokens * prices.per_output_token
            )
            if token_counts
            else 0.0
        )
        self.__estimated_sample_size = max(self.__estimated_sample_size, sample_size)
        self.__logger.info(
            f"Upper-bound cost of {sample_size} samples with {self.__model}: ${cost:.2f}"
        )
        return cost

    def warn_if_unestimated(self, sample_size: int) -> None:
        """Warn if no estimate covers generating this sample size.

        Args:
            sample_size (int): Number of samples about to be generated

        """
        if sample_size > self.__estimated_sample_size:
            self.__logger.warning(
                f"No cost estimate covers {sample_size} samples with {self.__model}; "
                "call estimate_cost first"
            )
