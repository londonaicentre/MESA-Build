from bs4 import BeautifulSoup, Tag
import markdown


class Assets:
    @staticmethod
    def markdown_to_text(source: str, remove_title: bool = True) -> str:
        html: str = markdown.markdown(source, extensions=["fenced_code"])
        soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        h1: Tag | None = soup.find("h1")
        if remove_title and h1:
            h1.decompose()
        return soup.get_text()
