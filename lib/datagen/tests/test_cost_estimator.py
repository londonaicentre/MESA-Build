import json
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from botocore.stub import Stubber
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_pricing import PricingClient
from pytest_mock import MockerFixture

from datagen.cost_estimator import CostEstimator, TokenPrices
from mesa_types import Document


class CostEstimatorFixture(CostEstimator):
    def prices(self) -> TokenPrices:
        return self._prices()

    def count_tokens(self, document: Document) -> int:
        return self._count_tokens(document)

    def get_runtime(self) -> BedrockRuntimeClient:
        return self._runtime

    def get_pricing(self) -> PricingClient:
        return self._pricing


@dataclass
class EstimateMocks:
    _prices: MagicMock
    _count_tokens: MagicMock


@pytest.fixture
def estimator(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> CostEstimatorFixture:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "foo")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "bar")
    return CostEstimatorFixture(
        getattr(request, "param", "eu.anthropic.foo-bar-1-2-v1:0"),
        "Foo Bar 1.2",
        "eu-west-2",
        "baz",
        100,
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document.model_validate(
            {
                "source": "foo",
                "content": f"bar{index}",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
        for index in range(3)
    ]


@pytest.fixture
def batch_price_list(request: pytest.FixtureRequest) -> list[str]:
    return [
        json.dumps(
            {
                "product": {
                    "attributes": {
                        "usagetype": f"EUW2-MP:EUW2_{token_type}TokenCount{scope}_Batch-Units",
                        "inferenceType": f"{token_type} tokens",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "foo": {
                            "priceDimensions": {
                                "bar": {
                                    "unit": getattr(request, "param", "1M tokens"),
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )
        for token_type, scope, price in [
            ("Input", "", "2.0"),
            ("Output", "", "8.0"),
            ("Input", "_Global", "1.0"),
            ("Output", "_Global", "4.0"),
        ]
    ]


@pytest.fixture
def pricing_stub(estimator: CostEstimatorFixture) -> Iterator[Stubber]:
    with Stubber(estimator.get_pricing()) as stubber:
        yield stubber


@pytest.fixture
def runtime_stub(estimator: CostEstimatorFixture) -> Iterator[Stubber]:
    with Stubber(estimator.get_runtime()) as stubber:
        yield stubber


@pytest.fixture
def create_anthropic_bedrock_batch_entry(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "datagen.cost_estimator.AWS.create_anthropic_bedrock_batch_entry",
        autospec=True,
        return_value={"recordId": "", "modelInput": {"foo": "bar"}},
    )


@pytest.fixture
def estimate_mocks(
    mocker: MockerFixture, estimator: CostEstimatorFixture
) -> EstimateMocks:
    return EstimateMocks(
        _prices=mocker.patch.object(
            estimator, "_prices", return_value=TokenPrices(0.01, 0.001)
        ),
        _count_tokens=mocker.patch.object(
            estimator,
            "_count_tokens",
            side_effect=lambda document: 100 * (int(document.content[3:]) + 1),
        ),
    )


@pytest.mark.parametrize(
    ("batch_price_list", "expected"),
    [
        ("1M tokens", TokenPrices(2e-06, 8e-06)),
        ("1K tokens", TokenPrices(2e-03, 8e-03)),
        ("10 tokens", TokenPrices(0.2, 0.8)),
    ],
    indirect=["batch_price_list"],
)
def test_prices_price_list_given_returns_per_token_prices(
    pricing_stub: Stubber,
    batch_price_list: list[str],
    estimator: CostEstimatorFixture,
    expected: TokenPrices,
) -> None:
    pricing_stub.add_response(
        "get_products",
        {"PriceList": batch_price_list},
        {
            "ServiceCode": "AmazonBedrockFoundationModels",
            "Filters": [
                {"Type": "TERM_MATCH", "Field": "model", "Value": "Foo Bar 1.2"},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": "eu-west-2"},
                {"Type": "TERM_MATCH", "Field": "feature", "Value": "Batch Inference"},
            ],
        },
    )
    assert estimator.prices() == pytest.approx(expected)


def test_prices_rate_missing_raises(
    pricing_stub: Stubber, batch_price_list: list[str], estimator: CostEstimatorFixture
) -> None:
    pricing_stub.add_response("get_products", {"PriceList": batch_price_list[:1]})
    with pytest.raises(ValueError, match="No batch token prices"):
        estimator.prices()


@pytest.mark.parametrize(
    "estimator",
    ["eu.anthropic.foo-bar-1-2-v1:0", "anthropic.foo-bar-1-2-v1:0"],
    indirect=True,
)
@pytest.mark.parametrize("batch_price_list", ["1 hour"], indirect=True)
def test_prices_unit_unrecognised_raises(
    pricing_stub: Stubber, batch_price_list: list[str], estimator: CostEstimatorFixture
) -> None:
    pricing_stub.add_response("get_products", {"PriceList": batch_price_list})
    with pytest.raises(ValueError, match="Unrecognised price unit: 1 hour"):
        estimator.prices()


def test_count_tokens_document_given_counts_with_base_model(
    runtime_stub: Stubber,
    create_anthropic_bedrock_batch_entry: MagicMock,
    estimator: CostEstimatorFixture,
    documents: list[Document],
) -> None:
    runtime_stub.add_response(
        "count_tokens",
        {"inputTokens": 42},
        {
            "modelId": "anthropic.foo-bar-1-2-v1:0",
            "input": {"invokeModel": {"body": '{"foo": "bar"}'}},
        },
    )
    assert estimator.count_tokens(documents[0]) == 42
    create_anthropic_bedrock_batch_entry.assert_called_once_with("", "baz", "bar0", 100)


@pytest.mark.parametrize(
    ("sample_size", "expected"),
    [(1, 1.1), (3, 6.3), (10, 6.3)],
)
def test_estimate_sample_size_given_returns_cost(
    estimate_mocks: EstimateMocks,
    estimator: CostEstimatorFixture,
    documents: list[Document],
    sample_size: int,
    expected: float,
) -> None:
    assert estimator.estimate(documents, sample_size) == pytest.approx(expected)


def test_estimate_many_documents_counts_spread_sample(
    estimate_mocks: EstimateMocks, estimator: CostEstimatorFixture
) -> None:
    assert estimator.estimate(
        [
            Document.model_validate(
                {
                    "source": "foo",
                    "content": f"bar{index}",
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
            for index in range(60)
        ],
        60,
    ) == pytest.approx(60 * (3000 * 0.01 + 100 * 0.001))
    assert [
        call.args[0].content for call in estimate_mocks._count_tokens.call_args_list
    ] == [f"bar{index * 60 // 50}" for index in range(50)]


def test_estimate_no_documents_returns_zero(
    estimate_mocks: EstimateMocks, estimator: CostEstimatorFixture
) -> None:
    assert estimator.estimate([], 5) == 0.0


@pytest.mark.parametrize(
    ("estimated_sizes", "sample_size", "warned"),
    [
        ([], 10, True),
        ([5], 10, True),
        ([10, 5], 10, False),
    ],
)
def test_warn_if_unestimated_sample_size_given_warns_when_uncovered(
    mocker: MockerFixture,
    estimate_mocks: EstimateMocks,
    estimator: CostEstimatorFixture,
    estimated_sizes: list[int],
    sample_size: int,
    warned: bool,
) -> None:
    for estimated_size in estimated_sizes:
        estimator.estimate([], estimated_size)
    warning: MagicMock = mocker.patch("logging.Logger.warning")
    estimator.warn_if_unestimated(sample_size)
    assert warning.called == warned
