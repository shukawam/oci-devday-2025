import requests, re
from bs4 import BeautifulSoup
import pandas as pd

def main():
    response = requests.get("https://www.oracle.com/jp/developer/events/dev-day/")
    soup = BeautifulSoup(response.text, "html.parser")

    titles = []
    r_titles = soup.find_all("button", {"class": "rc51title"})
    for title in r_titles:
        # 改行コード削除
        text = title.text.replace("\n", "")
        # タイトルの前にあるセッションIDを削除
        text = re.sub(r"\[.+?\]", "", text).strip()
        # 休憩の文字列を削除
        text = re.sub(r"休憩", "", text).strip()
        titles.append(text)

    abstracts = []
    r_abstracts = soup.find_all("div", {"class": "rc51desc"})
    for abstract in r_abstracts:
        # 改行コード削除
        text = abstract.text.replace("\n", "")
        abstracts.append(text)

    df = pd.DataFrame({"title": titles, "abstract": abstracts})
    df.to_csv("./data/sessions.csv", index=False)

if __name__ == "__main__":
    main()
