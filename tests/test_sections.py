from sentinel.ingest.sections import Sections

TEN_K = (
    "Table of Contents\n"
    "Item 1. Business 3\n"
    "Item 1A. Risk Factors 9\n"
    "Item 7. Management's Discussion 30\n"
    "PART I\n"
    "Item 1. Business\n"
    "We design phones.\n"
    "Item\xa01A.\xa0Risk Factors\n"  # SEC HTML: non-breaking spaces
    "Suppliers are concentrated.\n"
    "ITEM 7. MANAGEMENT'S DISCUSSION\n"
    "Revenue grew. See Item 1A above.\n"
)


def test_body_headings_win_over_table_of_contents() -> None:
    s = Sections(TEN_K)
    assert s.at(TEN_K.index("We design")) == "Item 1"
    assert s.at(TEN_K.index("Suppliers")) == "Item 1A"
    assert s.at(TEN_K.index("Revenue")) == "Item 7"  # mid-line "Item 1A" is not a heading


def test_text_before_first_heading_is_preamble() -> None:
    assert Sections(TEN_K).at(TEN_K.index("Item 1. Business 3")) == "preamble"


def test_8k_items() -> None:
    text = (
        "FORM 8-K\nItem 2.02 Results of Operations\nRevenue rose.\n"
        "Item 9.01 Exhibits\nEx 99.1\n"
    )
    s = Sections(text)
    assert s.at(text.index("Revenue")) == "Item 2.02"
    assert s.at(text.index("Ex 99.1")) == "Item 9.01"
