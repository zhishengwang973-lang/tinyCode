class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, sku: str, count: int) -> None:
        self._items[sku] = self._items.get(sku, 0) + count

    def take(self, sku: str, count: int) -> None:
        self._items[sku] = self._items.get(sku, 0) - count

    def count(self, sku: str) -> int:
        return self._items.get(sku, 0)
