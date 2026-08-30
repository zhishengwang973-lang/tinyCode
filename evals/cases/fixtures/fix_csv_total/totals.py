def total_amount(csv_text: str) -> int:
    return sum(int(line.split(",")[1]) for line in csv_text.splitlines())
