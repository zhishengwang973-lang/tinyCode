class PaymentService:
    def charge(self, amount: int) -> bool:
        return amount > 0

    def refund(self, payment_id: str) -> bool:
        return bool(payment_id)
