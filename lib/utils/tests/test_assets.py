from utils.assets import Assets


def test_markdown_to_text() -> None:
    source: str = '# foo\n**bar**\n```json\n{"baz":\n{"qux":"quux"}}\n```'
    assert (
        Assets.markdown_to_text(source, False) == 'foo\nbar\n{"baz":\n{"qux":"quux"}}\n'
    )
    Assets.markdown_to_text(source) == 'bar\n{"baz":\n{"qux":"quux"}}\n'
