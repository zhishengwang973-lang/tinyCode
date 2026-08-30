import unittest
from inventory import Inventory


class InventoryTests(unittest.TestCase):
    def test_take_cannot_make_stock_negative(self):
        inventory = Inventory()
        inventory.add("book", 2)
        with self.assertRaises(ValueError):
            inventory.take("book", 3)
        self.assertEqual(2, inventory.count("book"))

    def test_remove_clears_sku(self):
        inventory = Inventory()
        inventory.add("book", 2)
        inventory.remove("book")
        self.assertEqual(0, inventory.count("book"))
